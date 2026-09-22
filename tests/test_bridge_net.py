from __future__ import annotations

import socket
import threading
import time

import pytest

from nscr_houdini_mcp.bridge import net


def test_the_lowest_free_port_in_the_range_is_taken() -> None:
    taken = {18100, 18101}
    assert net.pick_port((18100, 18199), is_free=lambda port: port not in taken) == 18102


def test_a_range_with_nothing_free_fails_rather_than_wandering_off() -> None:
    with pytest.raises(net.PortUnavailable):
        net.pick_port((18100, 18102), is_free=lambda port: False)


@pytest.mark.parametrize("port_range", [(0, 10), (18199, 18100), (-1, 5)])
def test_a_range_that_is_not_a_range_is_refused(port_range: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        net.pick_port(port_range, is_free=lambda port: True)


def test_a_bound_port_is_not_free_and_is_free_again_afterwards() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind((net.LOOPBACK, 0))
        held.listen(1)
        port = held.getsockname()[1]
        assert net.port_is_free(port) is False
    assert net.port_is_free(port) is True


def test_this_machine_does_not_list_its_loopback_as_an_outside_address() -> None:
    for address in net.outward_addresses():
        assert not address.startswith("127.")
        assert address not in ("::1", "0.0.0.0", "::")
        assert not address.lower().startswith("fe80")


def test_a_name_lookup_that_does_not_answer_is_not_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()

    def stuck(*_args: object, **_kwargs: object) -> list:
        release.wait(30.0)
        raise OSError("the resolver gave up")

    monkeypatch.setattr(net.socket, "getaddrinfo", stuck)
    started = time.monotonic()
    try:
        assert net.own_name_addresses(timeout_s=0.2) == []
        net.outward_addresses()
    finally:
        release.set()
    assert time.monotonic() - started < 2 * net.NAME_LOOKUP_TIMEOUT_S + 1.0


def test_a_port_bound_to_loopback_answers_there_and_nowhere_else() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind((net.LOOPBACK, 0))
        held.listen(1)
        port = held.getsockname()[1]
        assert net.can_connect(net.LOOPBACK, port) is True
        assert net.reachable_from_outside(port) == []


def test_an_address_that_answers_is_reported_back() -> None:
    answered = net.reachable_from_outside(
        18100,
        addresses=["10.0.0.5", "10.0.0.6"],
        connect=lambda address, port: address == "10.0.0.6",
    )
    assert answered == ["10.0.0.6"]


def test_an_address_with_nothing_listening_can_be_bound() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind((net.LOOPBACK, 0))
        held.listen(1)
        port = held.getsockname()[1]
        assert net.addresses_holding_port(port, addresses=[net.LOOPBACK]) == [net.LOOPBACK]
        assert net.addresses_holding_port(port) == []


def test_binding_probes_leave_nothing_behind() -> None:
    port = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind((net.LOOPBACK, 0))
        port = held.getsockname()[1]
    assert net.addresses_holding_port(port, addresses=[net.LOOPBACK]) == []
    assert net.port_is_free(port) is True
