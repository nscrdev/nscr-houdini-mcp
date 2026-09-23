"""Proving who sent a request without ever sending the secret.

A token in a header is only safe while the port belongs to the bridge that
minted it. It does not always: a Houdini that crashes leaves its session file
behind, and anything can take the freed port and collect whatever the next
caller sends. So the token is never put on the wire. Each request carries a
signature over what it is asking for, and each answer carries a signature over
what it replied, both keyed by the token.

What that buys:

- Something sitting on the port learns a session id, a timestamp, a nonce and
  a signature that fits that one request and nothing else.
- A caller can tell a real bridge from something squatting on its port,
  because only the real one can sign the answer.
- A signature caught in flight cannot be sent again: the nonce is remembered
  for as long as the timestamp window lasts, and outside that window the
  timestamp alone refuses it.

Nothing here imports Houdini, a transport or the store.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import threading
import time
from collections.abc import Mapping

SESSION_HEADER = "x-nscr-mcp-session"
TIMESTAMP_HEADER = "x-nscr-mcp-timestamp"
NONCE_HEADER = "x-nscr-mcp-nonce"
SIGNATURE_HEADER = "x-nscr-mcp-signature"

REQUEST_VERSION = "nscr-mcp-request-v1"
RESPONSE_VERSION = "nscr-mcp-response-v1"

NONCE_BYTES = 16

# How far a caller's clock may be from the bridge's before a request is
# refused, and so also how long a nonce has to be remembered.
DEFAULT_SKEW_S = 120

# A bound on the remembered nonces, so a flood of signed requests cannot grow
# the table without end.
MAX_NONCES = 20000


class SignatureRefused(Exception):
    """The request did not prove it came from the holder of the token."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FloodGuard(SignatureRefused):
    """A signed request arrived while the nonce table was full.

    Only a request whose signature matched gets this far, so the caller holds
    the token and may be told plainly what happened: how many requests the
    window holds, how long the window is and how long until room comes free.
    """

    def __init__(self, *, limit: int, window_s: int, wait_s: int) -> None:
        super().__init__(f"{limit} signed requests arrived inside {window_s} seconds")
        self.limit = limit
        self.window_s = window_s
        self.wait_s = wait_s


def body_digest(body: bytes) -> str:
    """Hex digest of a request or response body, empty body included."""
    return hashlib.sha256(body).hexdigest()


def new_nonce() -> str:
    """A value used once, to bind a signature to a single request."""
    return secrets.token_hex(NONCE_BYTES)


def request_signature(
    token: str,
    *,
    method: str,
    path: str,
    session_id: str,
    timestamp: str,
    nonce: str,
    body: bytes,
) -> str:
    """Signature over everything the bridge will act on."""
    material = "\n".join(
        (
            REQUEST_VERSION,
            method.upper(),
            path,
            session_id,
            str(timestamp),
            nonce,
            body_digest(body),
        )
    )
    return _sign(token, material)


def response_signature(token: str, *, nonce: str, status: int, body: bytes) -> str:
    """Signature over one answer, tied to the nonce that asked for it."""
    material = "\n".join((RESPONSE_VERSION, nonce, str(status), body_digest(body)))
    return _sign(token, material)


def sign_request(
    token: str,
    *,
    method: str,
    path: str,
    session_id: str,
    body: bytes,
    now: float | None = None,
) -> dict[str, str]:
    """The headers one signed request needs."""
    timestamp = str(int(time.time() if now is None else now))
    nonce = new_nonce()
    return {
        SESSION_HEADER: session_id,
        TIMESTAMP_HEADER: timestamp,
        NONCE_HEADER: nonce,
        SIGNATURE_HEADER: request_signature(
            token,
            method=method,
            path=path,
            session_id=session_id,
            timestamp=timestamp,
            nonce=nonce,
            body=body,
        ),
    }


def equal(left: str, right: str) -> bool:
    """Constant time compare that survives an odd shaped value."""
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    try:
        return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
    except (TypeError, ValueError):
        return False


def _sign(token: str, material: str) -> str:
    return hmac.new(token.encode("utf-8"), material.encode("utf-8"), hashlib.sha256).hexdigest()


class NonceLog:
    """The nonces seen inside the timestamp window, so none is used twice."""

    def __init__(self, *, window_s: int = DEFAULT_SKEW_S, limit: int = MAX_NONCES) -> None:
        self.window_s = window_s
        self.limit = limit
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def claim(self, nonce: str, now: float) -> bool:
        """Take a nonce, or say it has already been used.

        Raises `FloodGuard` when the table is full: something is sending more
        signed requests than the window can remember, and refusing is the
        safe answer. The refusal says how long until the oldest nonce ages
        out and a request can be taken again.
        """
        with self._lock:
            self._forget(now)
            if nonce in self._seen:
                return False
            if len(self._seen) >= self.limit:
                oldest = min(self._seen.values())
                wait = max(1, math.ceil(oldest + self.window_s - now))
                raise FloodGuard(limit=self.limit, window_s=self.window_s, wait_s=wait)
            self._seen[nonce] = now
            return True

    def _forget(self, now: float) -> None:
        cutoff = now - self.window_s
        stale = [nonce for nonce, seen in self._seen.items() if seen < cutoff]
        for nonce in stale:
            del self._seen[nonce]

    def __len__(self) -> int:
        return len(self._seen)


class Verifier:
    """Checks signed requests for one session, and signs the answers."""

    def __init__(
        self,
        token: str,
        session_id: str,
        *,
        skew_s: int = DEFAULT_SKEW_S,
        nonces: NonceLog | None = None,
    ) -> None:
        self._token = token
        self._session_id = session_id
        self.skew_s = skew_s
        # An empty log has no length and so reads as false: test for None.
        self.nonces = nonces if nonces is not None else NonceLog(window_s=skew_s)

    def nonce_of(self, headers: Mapping[str, str]) -> str:
        """The nonce a request carried, for signing the answer to it."""
        return headers.get(NONCE_HEADER, "")

    def check(
        self,
        headers: Mapping[str, str],
        *,
        method: str,
        path: str,
        body: bytes,
        now: float | None = None,
    ) -> None:
        """Accept a request, or raise `SignatureRefused`.

        The reason is kept for the bridge's own log. What goes back to the
        caller says only that the request was refused.
        """
        moment = time.time() if now is None else now
        session_id = headers.get(SESSION_HEADER)
        timestamp = headers.get(TIMESTAMP_HEADER)
        nonce = headers.get(NONCE_HEADER)
        signature = headers.get(SIGNATURE_HEADER)
        if not (session_id and timestamp and nonce and signature):
            raise SignatureRefused("a signing header is missing")
        if not equal(session_id, self._session_id):
            raise SignatureRefused("addressed to another session")
        try:
            sent_at = int(timestamp)
        except ValueError:
            raise SignatureRefused("the timestamp is not a number") from None
        if abs(moment - sent_at) > self.skew_s:
            raise SignatureRefused("the timestamp is outside the window")
        if len(nonce) < 8 or len(nonce) > 128:
            raise SignatureRefused("the nonce is not the right length")
        expected = request_signature(
            self._token,
            method=method,
            path=path,
            session_id=self._session_id,
            timestamp=timestamp,
            nonce=nonce,
            body=body,
        )
        if not equal(signature, expected):
            raise SignatureRefused("the signature does not match")
        # Last, so a wrong signature cannot spend a nonce. A full table raises
        # `FloodGuard` from here, which is a `SignatureRefused` as well.
        if not self.nonces.claim(nonce, moment):
            raise SignatureRefused("the nonce has been used")

    def sign_answer(self, nonce: str, status: int, body: bytes) -> str:
        """Sign one answer so the caller can tell this bridge from a squatter."""
        return response_signature(self._token, nonce=nonce, status=status, body=body)
