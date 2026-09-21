"""Picking a port and proving that only loopback can reach it.

The web server binds every interface unless it is told otherwise, so the bind
address is a setting the bridge must pass and then check. The check is a
connection attempt from this machine to its own outside addresses: if one of
them answers on the bridge port, the bind did not do what it was told and the
bridge stops rather than serving the network.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Sequence

LOOPBACK = "127.0.0.1"

# A private range well above the ports Houdini and its help server use.
DEFAULT_PORT_RANGE = (18100, 18199)

# Documentation range. Connecting a UDP socket to it sends nothing but makes
# the system name the interface it would route through, which is how the
# machine's own outward address is found without a name lookup.
ROUTE_PROBE_ADDRESS = ("192.0.2.1", 9)

CONNECT_TIMEOUT_S = 0.4


class PortUnavailable(Exception):
    """Every port in the range was taken."""


def port_is_free(port: int, *, address: str = LOOPBACK) -> bool:
    """Whether a port can be bound right now.

    No address reuse flag: the question is whether the port is free, not
    whether it can be shared.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((address, port))
        except OSError:
            return False
    return True


def pick_port(
    port_range: tuple[int, int] = DEFAULT_PORT_RANGE,
    *,
    is_free: Callable[[int], bool] | None = None,
) -> int:
    """Lowest free port in the range.

    Another process can take the port between this answer and the bind, so the
    server is also given the top of the range and walks up from here itself.
    """
    start, end = port_range
    if start < 1 or end < start:
        raise ValueError(f"not a port range: {port_range}")
    free = port_is_free if is_free is None else is_free
    for port in range(start, end + 1):
        if free(port):
            return port
    raise PortUnavailable(f"no free port between {start} and {end}")


def outward_addresses() -> list[str]:
    """This machine's own non loopback addresses, as far as it can tell.

    Used to prove a port is unreachable from them. Link local addresses are
    left out because reaching one needs a scope id, so a failed connection to
    one would prove nothing.
    """
    found: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(ROUTE_PROBE_ADDRESS)
            found.append(probe.getsockname()[0])
        except OSError:
            pass
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, type=socket.SOCK_STREAM)
    except OSError:
        infos = []
    for info in infos:
        found.append(info[4][0])
    keep: list[str] = []
    for address in found:
        plain = address.split("%", 1)[0]
        if plain.startswith("127.") or plain in ("::1", "0.0.0.0", "::"):
            continue
        if plain.lower().startswith("fe80"):
            continue
        if plain not in keep:
            keep.append(plain)
    return keep


def can_connect(address: str, port: int, *, timeout_s: float = CONNECT_TIMEOUT_S) -> bool:
    """Whether a TCP connection to this address and port is accepted."""
    try:
        infos = socket.getaddrinfo(address, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for family, kind, proto, _canonical, sockaddr in infos:
        with socket.socket(family, kind, proto) as probe:
            probe.settimeout(timeout_s)
            try:
                probe.connect(sockaddr)
            except OSError:
                continue
        return True
    return False


def reachable_from_outside(
    port: int,
    *,
    addresses: Sequence[str] | None = None,
    timeout_s: float = CONNECT_TIMEOUT_S,
    connect: Callable[[str, int], bool] | None = None,
) -> list[str]:
    """The machine's own outside addresses that answer on this port.

    An empty list is the wanted answer. A firewall can also produce an empty
    list, so this proves the bind is not obviously wrong rather than proving
    the port is unreachable from every machine on the network.
    """
    attempt = connect or (lambda host, number: can_connect(host, number, timeout_s=timeout_s))
    candidates = outward_addresses() if addresses is None else list(addresses)
    return [address for address in candidates if attempt(address, port)]


def addresses_holding_port(port: int, *, addresses: Sequence[str] | None = None) -> list[str]:
    """This machine's outside addresses where the port is already taken.

    Binding an address and port that something is already listening on fails,
    so a bind that succeeds says nothing is listening there. It needs no
    outside package and, unlike a connection attempt, a firewall cannot make
    it look better than it is. Another program holding the same port on that
    address would also show up here, which is why this reports addresses
    rather than deciding anything.
    """
    candidates = outward_addresses() if addresses is None else list(addresses)
    held = []
    for address in candidates:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((address, port))
            except OSError:
                held.append(address)
    return held
