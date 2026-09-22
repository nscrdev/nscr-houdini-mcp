"""The smallest client that can talk to a bridge.

It posts JSON to one of the bridge's two paths and reads JSON back. Three
things it will not do:

- It never sends the token. Each request carries a signature made with it, so
  whatever is on the port learns nothing it could use again.
- It never trusts an answer it cannot check. Every reply is signed with the
  same token, and a reply that does not match is treated as coming from
  something other than the bridge.
- It never talks to a session whose process has gone. A session file outlives
  a crash, and the port in it is free for anything to take. A call addressed
  to a session id that is not there is refused here, before anything is sent,
  with the id of the session now answering to the same name. The dead process
  cannot say any of that for itself, and the live one must never be handed
  work that was written for its predecessor.

It sends no `Origin` and no `Referer`, which is what lets the bridge refuse
anything that does.

Retries. A read may be sent again as often as the caller likes. A call that
changes the scene is sent again only when it carries an operation id, and only
with the same id, because that is what makes the second send a receipt lookup
in the bridge rather than the same work done twice. One retry, on a lost reply
alone: the connection closed or the read ran out of time, which are the cases
where the work may well have happened and the answer never arrived. A reply
that arrived and said something is never retried, whatever it said.

Standard library only: the same module is imported inside Houdini.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import registry, signing
from nscr_houdini_mcp.bridge.net import LOOPBACK
from nscr_houdini_mcp.bridge.serving import CALL_PATH, HEALTH_PATH, JSON_TYPE

DEFAULT_TIMEOUT_S = 10.0

# How much longer than the bridge budgets this end waits on the socket.
SOCKET_MARGIN_S = 10.0


class BridgeUnreachable(Exception):
    """Nothing answered on that port."""


class BridgeNotAuthentic(Exception):
    """Something answered, but it could not prove it is the bridge."""


class SessionGone(Exception):
    """The process that owned this session file is not there any more."""


class SessionDead(SessionGone):
    """That session id belonged to a process that has gone.

    It carries what a caller needs to carry on: the name the dead session
    answered to, and the id of the session answering to that name now, when
    there is one. Nothing was sent.
    """

    CODE = "SESSION_DEAD"

    def __init__(
        self,
        session_id: str,
        *,
        alias: str | None = None,
        live_session_id: str | None = None,
    ) -> None:
        super().__init__(f"session {session_id} is not there any more")
        self.session_id = session_id
        self.alias = alias
        self.live_session_id = live_session_id

    def details(self) -> dict[str, Any]:
        """The recovery data, shaped like the details of a bridge error."""
        return {
            "code": self.CODE,
            "session_id": self.session_id,
            "alias": self.alias,
            "live_session_id": self.live_session_id,
            "hint": (
                "address the session answering to that name"
                if self.live_session_id
                else "start a session, then send the call again"
            ),
        }


class Answer(NamedTuple):
    """One answer: the status, the decoded body, the headers and the bytes.

    The bytes are kept because the signature is over exactly what arrived, and
    re-encoding the decoded body would not give the same text back.
    """

    status: int
    payload: Any
    headers: dict[str, str]
    raw: bytes = b""


class Session(NamedTuple):
    """Where one bridge is and what proves a request came from its owner."""

    session_id: str
    token: str
    port: int
    address: str = LOOPBACK

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> Session:
        return cls(
            session_id=str(entry["session_id"]),
            token=str(entry["token"]),
            port=int(entry["port"]),
            address=str(entry.get("address") or LOOPBACK),
        )

    @classmethod
    def open(cls, home: Path, handle: str, *, store_path: Path | None = None) -> Session:
        """Find a live session by id or alias, or say what became of it.

        A handle that names a session this machine has seen before, and whose
        process is gone, raises `SessionDead` with the id of whatever answers
        to the same name now. A handle nothing has ever heard of raises
        `SessionGone`.
        """
        home = Path(home)
        entry = registry.find_entry(home, handle)
        if entry is not None:
            return cls.from_entry(entry)
        known = _remembered(home, handle, store_path)
        if known is None:
            raise SessionGone(f"no live session {handle}")
        alias, live_id = known
        raise SessionDead(handle, alias=alias, live_session_id=live_id)


def _remembered(home: Path, handle: str, store_path: Path | None) -> tuple[str, str | None] | None:
    """What the coordination store remembers about a handle that is not live.

    Returns the name that handle answered to and the id of the live session
    under that name, or nothing when the store has never heard of it. The
    store is asked to mark the sessions whose processes are gone first, so a
    session that crashed does not hold its name against its successor.
    """
    path = Path(store_path) if store_path is not None else Path(home) / store_module.STORE_FILE_NAME
    if not path.exists():
        return None
    try:
        with store_module.Store(path) as store:
            store.reclaim_sessions()
            record = store.get_session(handle)
            if record is None:
                return None
            live = store.resolve_session(record.alias)
            live_id = None if live is None or live.session_id == handle else live.session_id
            return record.alias, live_id
    except store_module.StoreError:
        return None


def new_operation_id() -> str:
    """One id for one mutating call, and for every retry of that same call."""
    return f"op-{uuid.uuid4().hex}"


def request(
    port: int,
    path: str,
    *,
    body: bytes = b"",
    headers: Mapping[str, str] | None = None,
    content_type: str = JSON_TYPE,
    address: str = LOOPBACK,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    method: str = "POST",
) -> Answer:
    """Send one request exactly as given, signing nothing and checking nothing.

    This is the low level way in, for probing a port. Ordinary work goes
    through `post`, which signs what it sends and checks what comes back.
    """
    built = urllib.request.Request(
        f"http://{address}:{port}{path}",
        data=body,
        method=method,
        headers={"Content-Type": content_type},
    )
    for name, value in (headers or {}).items():
        built.add_header(name, value)
    try:
        with urllib.request.urlopen(built, timeout=timeout_s) as answer:  # noqa: S310
            raw = answer.read()
            return Answer(answer.status, _decode(raw), _headers(answer), raw)
    except urllib.error.HTTPError as error:
        raw = error.read()
        return Answer(error.code, _decode(raw), _headers(error), raw)
    except (urllib.error.URLError, OSError) as error:
        raise BridgeUnreachable(f"{address}:{port} did not answer: {error}") from error


def post(
    session: Session,
    path: str,
    payload: Mapping[str, Any] | None = None,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    verify: bool = True,
) -> Answer:
    """Send one signed request and check that the bridge signed the answer."""
    body = json.dumps(payload if payload is not None else {}).encode("utf-8")
    signed = signing.sign_request(
        session.token,
        method="POST",
        path=path,
        session_id=session.session_id,
        body=body,
    )
    signed.update(headers or {})
    answer = request(
        session.port,
        path,
        body=body,
        headers=signed,
        address=session.address,
        timeout_s=timeout_s,
    )
    if verify:
        _check_answer(session, signed[signing.NONCE_HEADER], answer)
    return answer


def health(session: Session, **rest: Any) -> Answer:
    """Ask a bridge whether it is alive."""
    return post(session, HEALTH_PATH, {}, **rest)


def call(
    session: Session,
    tool: str,
    *,
    arguments: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    scene_epoch: int | None = None,
    operation_id: str | None = None,
    wait_s: float | None = None,
    timeout_s: float | None = None,
    skip_if_busy: bool | None = None,
    http_timeout_s: float | None = None,
    retry_lost_reply: bool = True,
    **rest: Any,
) -> Answer:
    """Send one request envelope.

    `wait_s` and `timeout_s` are the bridge's budgets: how long the call may
    wait for its turn, and how long it may wait for work that is running. A
    call that names no `wait_s` waits one second and is then told the session
    is busy, so a caller that means to queue behind a long running call has to
    ask for a longer wait.
    `http_timeout_s` is how long this end waits on the socket. It defaults to
    a little more than both, because a client that gives up before the bridge
    answers learns nothing and leaves the work running.

    A lost reply is sent once more when, and only when, the call carries an
    `operation_id`. The second send carries the same id, so the bridge answers
    it from the receipt the first send took rather than doing the work again.
    Without an id there is no retry: repeating a mutation blind is how one
    request becomes two nodes.
    """
    envelope: dict[str, Any] = {"tool": tool, "arguments": dict(arguments or {})}
    if session_id is not None:
        envelope["session_id"] = session_id
    if scene_epoch is not None:
        envelope["scene_epoch"] = scene_epoch
    if operation_id is not None:
        envelope["operation_id"] = operation_id
    if wait_s is not None:
        envelope["wait_s"] = wait_s
    if timeout_s is not None:
        envelope["timeout_s"] = timeout_s
    if skip_if_busy is not None:
        envelope["skip_if_busy"] = skip_if_busy
    if http_timeout_s is None:
        http_timeout_s = max(
            DEFAULT_TIMEOUT_S, (wait_s or 0.0) + (timeout_s or 0.0) + SOCKET_MARGIN_S
        )
    rest.setdefault("timeout_s", http_timeout_s)
    try:
        return post(session, CALL_PATH, envelope, **rest)
    except BridgeUnreachable:
        if not (retry_lost_reply and operation_id):
            raise
    return post(session, CALL_PATH, envelope, **rest)


def mutate(
    session: Session,
    tool: str,
    *,
    operation_id: str | None = None,
    **rest: Any,
) -> Answer:
    """Send one call that changes the scene, safe to lose the answer to.

    It mints an operation id when the caller passes none, so the call is one
    the bridge can recognise if it ever arrives twice.
    """
    return call(session, tool, operation_id=operation_id or new_operation_id(), **rest)


def _check_answer(session: Session, nonce: str, answer: Answer) -> None:
    """Refuse an answer that the holder of the token did not sign."""
    given = answer.headers.get(signing.SIGNATURE_HEADER)
    if not given:
        raise BridgeNotAuthentic(f"{session.address}:{session.port} signed no answer")
    expected = signing.response_signature(
        session.token, nonce=nonce, status=answer.status, body=answer.raw
    )
    if not signing.equal(given, expected):
        raise BridgeNotAuthentic(f"{session.address}:{session.port} is not this session")


def _headers(answer: Any) -> dict[str, str]:
    """Response headers, names lowercased so a check cannot miss one."""
    return {str(name).lower(): str(value) for name, value in answer.headers.items()}


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text
