from __future__ import annotations

import time

import pytest

from nscr_houdini_mcp.bridge import signing


def sign(token: str, session_id: str, **overrides: object) -> dict[str, str]:
    fields = {"method": "POST", "path": "/nscr-mcp/call", "body": b'{"tool": "bridge.ping"}'}
    fields.update(overrides)
    return signing.sign_request(token, session_id=session_id, **fields)  # type: ignore[arg-type]


def test_a_signed_request_is_accepted_once(monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = signing.Verifier("token", "session")
    body = b'{"tool": "bridge.ping"}'
    headers = sign("token", "session", body=body)
    verifier.check(headers, method="POST", path="/nscr-mcp/call", body=body)
    with pytest.raises(signing.SignatureRefused):
        verifier.check(headers, method="POST", path="/nscr-mcp/call", body=body)


def test_the_token_is_never_one_of_the_headers() -> None:
    headers = sign("a-very-secret-token", "session")
    assert "a-very-secret-token" not in "".join(headers.values())
    assert set(headers) == {
        signing.SESSION_HEADER,
        signing.TIMESTAMP_HEADER,
        signing.NONCE_HEADER,
        signing.SIGNATURE_HEADER,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"method": "PUT"},
        {"path": "/nscr-mcp/health"},
        {"body": b'{"tool": "bridge.other"}'},
    ],
)
def test_a_signature_covers_the_method_the_path_and_the_body(change: dict) -> None:
    verifier = signing.Verifier("token", "session")
    body = b'{"tool": "bridge.ping"}'
    headers = sign("token", "session", body=body)
    asked = {"method": "POST", "path": "/nscr-mcp/call", "body": body}
    asked.update(change)
    with pytest.raises(signing.SignatureRefused):
        verifier.check(headers, **asked)  # type: ignore[arg-type]


def test_another_token_and_another_session_are_both_refused() -> None:
    verifier = signing.Verifier("token", "session")
    body = b"{}"
    for headers in (sign("other", "session", body=body), sign("token", "other", body=body)):
        with pytest.raises(signing.SignatureRefused):
            verifier.check(headers, method="POST", path="/nscr-mcp/call", body=body)


@pytest.mark.parametrize(
    "missing",
    [
        signing.SESSION_HEADER,
        signing.TIMESTAMP_HEADER,
        signing.NONCE_HEADER,
        signing.SIGNATURE_HEADER,
    ],
)
def test_a_request_missing_a_signing_header_is_refused(missing: str) -> None:
    verifier = signing.Verifier("token", "session")
    headers = sign("token", "session")
    del headers[missing]
    with pytest.raises(signing.SignatureRefused):
        verifier.check(headers, method="POST", path="/nscr-mcp/call", body=b"{}")


@pytest.mark.parametrize("offset", [-10_000, 10_000])
def test_a_timestamp_outside_the_window_is_refused(offset: int) -> None:
    verifier = signing.Verifier("token", "session")
    headers = sign("token", "session", now=time.time() + offset)
    with pytest.raises(signing.SignatureRefused):
        verifier.check(headers, method="POST", path="/nscr-mcp/call", body=b"{}")


def test_a_timestamp_that_is_not_a_number_is_refused() -> None:
    verifier = signing.Verifier("token", "session")
    headers = sign("token", "session")
    headers[signing.TIMESTAMP_HEADER] = "yesterday"
    with pytest.raises(signing.SignatureRefused):
        verifier.check(headers, method="POST", path="/nscr-mcp/call", body=b"{}")


def test_a_wrong_signature_does_not_spend_the_nonce() -> None:
    verifier = signing.Verifier("token", "session")
    body = b"{}"
    headers = sign("token", "session", body=body)
    broken = dict(headers)
    broken[signing.SIGNATURE_HEADER] = "0" * 64
    with pytest.raises(signing.SignatureRefused):
        verifier.check(broken, method="POST", path="/nscr-mcp/call", body=body)
    verifier.check(headers, method="POST", path="/nscr-mcp/call", body=body)


def test_remembered_nonces_are_let_go_once_the_window_has_passed() -> None:
    log = signing.NonceLog(window_s=10)
    assert log.claim("one", 100.0) is True
    assert log.claim("one", 105.0) is False
    assert log.claim("one", 200.0) is True
    assert len(log) == 1


def test_the_nonce_table_does_not_grow_without_end() -> None:
    log = signing.NonceLog(window_s=1000, limit=3)
    for index in range(3):
        assert log.claim(f"n{index}", 100.0) is True
    with pytest.raises(signing.FloodGuard):
        log.claim("n3", 100.0)
    assert len(log) == 3


def test_a_full_table_says_the_limit_the_window_and_how_long_to_wait() -> None:
    log = signing.NonceLog(window_s=120, limit=2)
    assert log.claim("a", 100.0) is True
    assert log.claim("b", 130.0) is True
    # A nonce already taken is still a replay, full table or not.
    assert log.claim("a", 150.0) is False
    with pytest.raises(signing.FloodGuard) as raised:
        log.claim("c", 150.5)
    flood = raised.value
    assert (flood.limit, flood.window_s) == (2, 120)
    # The oldest nonce ages out at 220, so room comes free in 69.5 seconds.
    assert flood.wait_s == 70
    assert isinstance(flood, signing.SignatureRefused)
    # Once the oldest has gone there is room again.
    assert log.claim("c", 220.5) is True


def test_the_verifier_raises_the_flood_guard_only_for_a_good_signature() -> None:
    verifier = signing.Verifier("token", "session", nonces=signing.NonceLog(limit=1))
    body = b"{}"
    first = sign("token", "session", body=body)
    verifier.check(first, method="POST", path="/nscr-mcp/call", body=body)
    forged = sign("other", "session", body=body)
    with pytest.raises(signing.SignatureRefused) as refused:
        verifier.check(forged, method="POST", path="/nscr-mcp/call", body=body)
    assert not isinstance(refused.value, signing.FloodGuard)
    second = sign("token", "session", body=body)
    with pytest.raises(signing.FloodGuard):
        verifier.check(second, method="POST", path="/nscr-mcp/call", body=body)


def test_an_answer_is_signed_for_the_nonce_that_asked_for_it() -> None:
    verifier = signing.Verifier("token", "session")
    signed = verifier.sign_answer("nonce", 200, b"body")
    assert signed == signing.response_signature("token", nonce="nonce", status=200, body=b"body")
    assert signed != signing.response_signature("token", nonce="other", status=200, body=b"body")
    assert signed != signing.response_signature("token", nonce="nonce", status=401, body=b"body")
    assert signed != signing.response_signature("token", nonce="nonce", status=200, body=b"other")
    assert signed != signing.response_signature("other", nonce="nonce", status=200, body=b"body")


@pytest.mark.parametrize("pair", [("a", "b"), ("a", ""), (7, "a"), ("a", None)])
def test_the_compare_survives_anything_it_is_handed(pair: tuple) -> None:
    assert signing.equal(*pair) is False


def test_a_nonce_is_never_the_same_twice() -> None:
    assert len({signing.new_nonce() for _ in range(100)}) == 100
