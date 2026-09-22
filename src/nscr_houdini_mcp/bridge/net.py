"""Picking a port and proving that only loopback can reach it.

The web server binds every interface unless it is told otherwise, so the bind
address is a setting the bridge must pass and then check.

Two checks, because neither is enough on its own:

- A bind attempt on each of this machine's own outside addresses. A bind that
  succeeds says nothing is listening there. This is the check the bridge
  starts on, because a firewall cannot make its answer look better than the
  truth.
- A connection attempt to the same addresses, which is what an outside caller
  would actually do.

What each proves, by system:

- Linux and macOS: no address reuse flag is set on the probe, so binding an
  address and port that something already holds fails. A bind that succeeds is
  real evidence that nothing is listening on that address.
- Windows: a second bind to the same address and port can succeed when the
  first socket asked for address reuse, and what the web server asked for is
  not visible from here. So on Windows a clean bind is weaker evidence, and
  the address each request actually arrived on is the check that settles it.

Neither check can say anything about an address this machine does not have.
When there is no outside address to test, nothing has been proven, and the
caller is told that rather than told it passed.
"""

from __future__ import annotations

import socket
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

LOOPBACK = "127.0.0.1"

# Names and addresses that mean "this machine, over the loopback interface".
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})

# A private range well above the ports Houdini and its help server use.
DEFAULT_PORT_RANGE = (18100, 18199)

# Documentation range. Connecting a UDP socket to it sends nothing but makes
# the system name the interface it would route through, which is how the
# machine's own outward address is found without a name lookup.
ROUTE_PROBE_ADDRESS = ("192.0.2.1", 9)

CONNECT_TIMEOUT_S = 0.4

# How long the machine's own name gets to resolve. Where no resolver knows the
# name, the lookup only ends when the resolver gives up, which on a macOS build
# machine takes about half a minute, and every bridge start would wait for it.
# The route probe has already found the address that matters by then.
NAME_LOOKUP_TIMEOUT_S = 1.0


class PortUnavailable(Exception):
    """Every port in the range was taken."""


def port_is_free(port: int, *, address: str = LOOPBACK) -> bool:
    """Whether the server could bind this port right now.

    The probe asks for address reuse exactly where the server does, so the
    answer is the one the server would get. On Linux and macOS that flag does
    not let a bind past a socket that is listening, so an answering port is
    still refused; what it does let past is a port whose old connections are
    draining, which the server can take and which would otherwise leave a
    whole range looking full after a busy session.

    On Windows the flag would let the bind past a live listener, so it is not
    set there and a draining port counts as taken.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if sys.platform != "win32":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
    found.extend(own_name_addresses())
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


def own_name_addresses(*, timeout_s: float = NAME_LOOKUP_TIMEOUT_S) -> list[str]:
    """What this machine's own name resolves to, or nothing if that is slow.

    The lookup runs on a daemon thread of its own, so a resolver that takes
    half a minute to give up is left to do so without holding up the caller.
    """
    found: list[str] = []

    def look_up() -> None:
        try:
            infos = socket.getaddrinfo(socket.gethostname(), None, type=socket.SOCK_STREAM)
        except OSError:
            return
        found.extend(str(info[4][0]) for info in infos)

    lookup = threading.Thread(target=look_up, name="nscr-mcp-name-lookup", daemon=True)
    lookup.start()
    lookup.join(timeout_s)
    return [] if lookup.is_alive() else found


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


def is_loopback(address: str | None) -> bool:
    """Whether an address is on this machine's loopback interface."""
    if not address:
        return False
    plain = str(address).split("%", 1)[0].strip().strip("[]")
    return plain.startswith("127.") or plain in ("::1", "0:0:0:0:0:0:0:1")


@dataclass(frozen=True)
class PrivacyProof:
    """What could be shown about who can reach a port, and what could not."""

    private: bool
    proven: bool
    tested: tuple[str, ...]
    reachable: tuple[str, ...]
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "private": self.private,
            "proven": self.proven,
            "tested": list(self.tested),
            "reachable": list(self.reachable),
            "note": self.note,
        }


def prove_loopback_only(port: int, *, addresses: Sequence[str] | None = None) -> PrivacyProof:
    """Test a port against this machine's own outside addresses.

    `private` false means something answered where nothing should. `proven`
    false means there was nothing to test against, which is not the same as a
    pass and is never reported as one.
    """
    candidates = tuple(outward_addresses() if addresses is None else addresses)
    if not candidates:
        return PrivacyProof(
            private=True,
            proven=False,
            tested=(),
            reachable=(),
            note="this machine has no address outside loopback to test against",
        )
    held = tuple(addresses_holding_port(port, addresses=candidates))
    answering = tuple(reachable_from_outside(port, addresses=candidates))
    busy = tuple(dict.fromkeys(held + answering))
    if busy:
        return PrivacyProof(
            private=False,
            proven=True,
            tested=candidates,
            reachable=busy,
            note="the port is taken or answering on an address outside loopback",
        )
    note = "nothing holds or answers on this machine's outside addresses"
    if sys.platform == "win32":
        note += ", though a bind can succeed here beside a socket that asked for reuse"
    return PrivacyProof(
        private=True,
        proven=sys.platform != "win32",
        tested=candidates,
        reachable=(),
        note=note,
    )
