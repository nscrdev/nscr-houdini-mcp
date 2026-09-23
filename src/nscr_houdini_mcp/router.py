"""Which Houdini session a call goes to, and the signed call that reaches it.

Resolution reads the coordination store every time, so nothing that matters is
kept in this process: another server may have started or lost a session since
the last call. The rules, in order:

1. `session` given: an id or an alias. A live session by that name is used. A
   session whose process has gone is refused with `SESSION_DEAD` and the id of
   the live session answering to the same alias now, if there is one; nothing
   is ever sent on to that successor, because work written for one process
   must not land in another. A session whose own port stopped answering is
   refused with `SESSION_UNRESPONSIVE`. A name nobody knows is
   `SESSION_UNKNOWN`, with the nearest aliases.
2. No `session` and exactly one live session: that one.
3. No `session` and several live: the config's `default_session` when it
   names one of them, otherwise `SESSION_AMBIGUOUS` with the candidates.
4. No live session at all: `NO_SESSION`.

The signed client for a session is kept per session id and dropped the moment
the session proves dead, unreachable or unable to sign, so the next call reads
the session afresh. A session id is never reused, so a kept client can only
ever reach the process it was made for.

Every call to a worker renews the worker's idle lease, when it is resolved and
again when its answer comes back, as the pool expects of any server that
routes to one. Reads count: a worker somebody is reading from is in use, so it
is kept warm on purpose.

Pacing. A call on its way to a session with a user interface waits here for
its turn under the pacer's rules, one call out at a time, a minimum pause
after each and a rate cap, so an agent in a loop cannot keep the artist's main
thread busy without a break. The wait comes out of the call's `wait_s`, and the
bridge is given what is left. A call whose turn is further off than that, or
that asked to be skipped when busy, is refused at once with `SESSION_BUSY` and
`retry_after_s`, and nothing is sent. The reply says how long the call waited
in `throttled_ms` and when it was let through in `admitted_at`. A worker is not
paced, unless the router is told to treat workers as sessions with an
interface, which is for tests. A cancel is never paced: it runs beside the
call it stops and never reaches the main thread.

The socket wait. When a call names no `timeout_s`, the bridge gives running
work its own default of a minute, so this end waits on the socket for the
queueing time plus that minute plus a margin. Waiting less would read a slow
answer as a lost one, and a lost change is sent again.

This module never imports `hou`.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from nscr_houdini_mcp import pool
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.bridge.dispatch import DEFAULT_TIMEOUT_S as BRIDGE_TIMEOUT_S
from nscr_houdini_mcp.bridge.dispatch import DEFAULT_WAIT_S as BRIDGE_WAIT_S
from nscr_houdini_mcp.bridge.errors import did_you_mean
from nscr_houdini_mcp.pacing import NoTurn, Pacer, throttled_ms
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.store import SessionRecord

LIVE_STATES = ("live", "busy")
DEAD_STATES = ("gone", "crashed")

# How long a health request may take. Health never waits on the scene, so a
# session that has not answered by then is not going to.
HEALTH_TIMEOUT_S = 2.0

# How much longer than the bridge's budgets this end waits on the socket.
SOCKET_MARGIN_S = client.SOCKET_MARGIN_S

# Codes in a reply that mean the kept client no longer fits the session.
STALE_CLIENT_CODES = frozenset({"SESSION_DEAD", "UNKNOWN_SESSION", "UNAUTHORIZED"})

# Bridge tools that never reach the main thread, so are never paced.
UNPACED_TOOLS = frozenset({"bridge.cancel"})

# Where a reply says how long pacing held it, and when, on the server's wall
# clock in seconds, pacing let it through.
THROTTLED_KEY = "throttled_ms"
ADMITTED_KEY = "admitted_at"


@dataclass(frozen=True)
class Target:
    """One resolved session: its row, and the client that can reach it."""

    record: SessionRecord
    session: client.Session
    houdini_version: str | None = None

    def __repr__(self) -> str:
        # The client holds the token, so it is never printed.
        return f"Target(session_id={self.record.session_id!r}, alias={self.record.alias!r})"

    @property
    def session_id(self) -> str:
        return self.record.session_id

    def trace(self) -> dict[str, Any]:
        return {
            "session_id": self.record.session_id,
            "alias": self.record.alias,
            "scene_epoch": self.record.scene_epoch,
        }


# Section: the rules, on plain rows


def row(record: SessionRecord) -> dict[str, Any]:
    """One session as a candidate a caller can pick from."""
    return {
        "session_id": record.session_id,
        "alias": record.alias,
        "kind": record.kind,
        "state": record.state,
        "hip_path": record.hip_path,
    }


def choose(
    records: Sequence[SessionRecord], handle: str | None, *, default: str | None = None
) -> SessionRecord:
    """Pick the session a call means, or raise `CallError` saying why not.

    `records` is every session the store knows, gone ones included, so a name
    that belonged to a process that has ended can be told apart from one that
    never existed.
    """
    live = [record for record in records if record.state in LIVE_STATES]
    if handle:
        return _named(records, live, handle.strip())
    if len(live) == 1:
        return live[0]
    if not live:
        waiting = [row(r) for r in records if r.state not in LIVE_STATES + DEAD_STATES]
        raise CallError(
            "NO_SESSION",
            "no Houdini session is live",
            details={"not_answering": waiting} if waiting else None,
        )
    if default:
        chosen = _match(live, default)
        if chosen is not None:
            return chosen
    details: dict[str, Any] = {"candidates": [row(record) for record in live]}
    if default:
        details["default_session"] = default
        details["default_live"] = False
    raise CallError(
        "SESSION_AMBIGUOUS",
        f"{len(live)} sessions are live and the call named none",
        details=details,
    )


def _named(
    records: Sequence[SessionRecord], live: Sequence[SessionRecord], handle: str
) -> SessionRecord:
    by_id = next((record for record in records if record.session_id == handle), None)
    if by_id is not None:
        return _usable(by_id, live)
    # The newest session under an alias is the one that name means now.
    by_alias = [record for record in records if record.alias == handle]
    by_alias.sort(key=lambda record: record.started_at, reverse=True)
    current = [record for record in by_alias if record.state not in DEAD_STATES]
    if current:
        return _usable(current[0], live)
    if by_alias:
        return _usable(by_alias[0], live)
    names = sorted(
        {r.alias for r in records if r.state not in DEAD_STATES} | {r.session_id for r in live}
    )
    raise CallError(
        "SESSION_UNKNOWN",
        f"no session answers to {handle}",
        details={
            "session": handle,
            "did_you_mean": did_you_mean(handle, names),
            "live": [row(record) for record in live],
        },
    )


def _usable(record: SessionRecord, live: Sequence[SessionRecord]) -> SessionRecord:
    if record.state in LIVE_STATES:
        return record
    if record.state in DEAD_STATES:
        raise dead(record.session_id, record.alias, live)
    raise CallError(
        "SESSION_UNRESPONSIVE",
        f"session {record.alias} is running but its port does not answer",
        details={
            "session_id": record.session_id,
            "alias": record.alias,
            "state": record.state,
            "live": [row(other) for other in live if other.session_id != record.session_id],
        },
    )


def dead(session_id: str, alias: str | None, live: Iterable[SessionRecord]) -> CallError:
    """`SESSION_DEAD`, naming the live session under the same alias, if any."""
    live = list(live)
    successor = next(
        (r for r in live if alias and r.alias == alias and r.session_id != session_id), None
    )
    details: dict[str, Any] = {
        "session_id": session_id,
        "alias": alias,
        "live_session_id": successor.session_id if successor else None,
        "live": [row(record) for record in live],
    }
    if successor is not None:
        hint = f"session {successor.session_id} answers to {alias} now; address it to go on"
        message = f"session {session_id} has ended; {alias} is a new process now"
    else:
        hint = None
        message = f"session {session_id} has ended"
    return CallError("SESSION_DEAD", message, hint=hint, details=details)


def _match(records: Iterable[SessionRecord], handle: str) -> SessionRecord | None:
    for record in records:
        if handle in (record.session_id, record.alias):
            return record
    return None


# Section: the router


def _open_store(path: Path) -> store_module.Store | None:
    """The store, or nothing when no session has ever written one.

    A server with nothing to route to creates no state folder of its own.
    """
    return store_module.Store(path) if path.is_file() else None


class Router:
    """Resolves sessions and sends signed calls to them."""

    def __init__(
        self,
        home: Path,
        *,
        default_session: str | None = None,
        store_path: Path | None = None,
        open_store: Callable[[Path], Any] = _open_store,
        open_session: Callable[..., client.Session] = client.Session.open,
        send: Callable[..., client.Answer] = client.call,
        ask_health: Callable[..., client.Answer] = client.health,
        renew_lease: Callable[[Any, str], Any] = pool.touch,
        pacer: Pacer | None = None,
        pace_workers: bool = False,
    ) -> None:
        self.home = Path(home)
        # No pacer means no pacing, which is what a router built by hand gets.
        self.pacer = pacer if pacer is not None else Pacer(min_pause_s=0, max_per_s=0)
        self.pace_workers = pace_workers
        self.default_session = default_session
        self.store_path = store_path or self.home / store_module.STORE_FILE_NAME
        self._open_store = open_store
        self._open_session = open_session
        self._send = send
        self._ask_health = ask_health
        self._renew_lease = renew_lease
        # Per session id: the signed client and the facts from its session file.
        self._clients: dict[str, tuple[client.Session, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # Section: resolution

    def resolve(self, handle: str | None) -> Target:
        """The session a call goes to, with a client that can reach it.

        Ended sessions are read only when a session is named, which is the one
        case where telling an ended session from an unknown name matters.
        """
        records = self._records(include_gone=bool(handle))
        self._prune(records)
        record = choose(records, handle, default=self.default_session)
        session, facts = self._client(record, records)
        if record.kind == "hython":
            self._renew(record.session_id)
        return Target(record, session, houdini_version=facts.get("houdini_version"))

    def reach(self, record: SessionRecord) -> Target:
        """A target for a row already read, for a look that is not a use.

        Nothing is renewed: listing the sessions is not using a worker, and a
        client that lists every few seconds must not keep every worker warm.
        """
        session, facts = self._client(record, [record])
        return Target(record, session, houdini_version=facts.get("houdini_version"))

    def records(self, *, include_gone: bool = True) -> list[SessionRecord]:
        """Every session the store knows, after marking the ones that have gone."""
        return self._records(include_gone=include_gone)

    @contextmanager
    def store(self, *, create: bool = False) -> Iterator[Any]:
        """The coordination store, or nothing when none has been written yet.

        `create` makes one when there is none, for a caller that is about to
        start the first session itself. Only a store that cannot be opened is
        `STORE_UNAVAILABLE` here. What the caller does with it raises what it
        raises.
        """
        try:
            opened = self._open_store(self.store_path)
            if opened is None and create:
                opened = store_module.Store(self.store_path)
        except (store_module.StoreError, sqlite3.Error, OSError) as error:
            raise CallError(
                "STORE_UNAVAILABLE",
                "the coordination store could not be read",
                details={"exception": type(error).__name__},
            ) from None
        if opened is None:
            yield None
            return
        with opened:
            yield opened

    def cached(self) -> list[str]:
        """The session ids a client is kept for."""
        with self._lock:
            return list(self._clients)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._clients.pop(session_id, None)

    def _prune(self, records: Sequence[SessionRecord]) -> None:
        """Drop the kept clients of sessions that are no longer live."""
        live = {r.session_id for r in records if r.state in LIVE_STATES}
        with self._lock:
            for session_id in [key for key in self._clients if key not in live]:
                del self._clients[session_id]
                self.pacer.forget(session_id)

    def _records(self, *, include_gone: bool = True) -> list[SessionRecord]:
        try:
            store = self._open_store(self.store_path)
            if store is None:
                return []
            with store:
                store.reclaim_sessions()
                return store.list_sessions(include_gone=include_gone)
        except (store_module.StoreError, sqlite3.Error, OSError) as error:
            raise CallError(
                "STORE_UNAVAILABLE",
                "the coordination store could not be read",
                details={"exception": type(error).__name__},
            ) from None

    def _client(
        self, record: SessionRecord, records: Sequence[SessionRecord]
    ) -> tuple[client.Session, dict[str, Any]]:
        with self._lock:
            kept = self._clients.get(record.session_id)
        if kept is not None:
            return kept
        live = [r for r in records if r.state in LIVE_STATES]
        try:
            session = self._open_session(self.home, record.session_id, store_path=self.store_path)
        except client.SessionDead as error:
            raise dead(record.session_id, record.alias, live) from error
        except client.SessionGone as error:
            # The row says live but the session file is not there: the
            # process ended cleanly a moment ago, or never finished starting.
            raise dead(record.session_id, record.alias, live) from error
        if session.deaf():
            raise CallError(
                "SESSION_UNRESPONSIVE",
                f"session {record.alias} said its own port was not answering",
                details={"session_id": record.session_id, "alias": record.alias},
            )
        kept = (session, session_facts(self.home, record.session_id))
        with self._lock:
            self._clients[record.session_id] = kept
        return kept

    def _renew(self, session_id: str) -> None:
        """Renew a worker's idle lease. A hython started by hand has none."""
        try:
            store = self._open_store(self.store_path)
            if store is None:
                return
            with store:
                self._renew_lease(store, session_id)
        except (pool.UnknownWorker, store_module.StoreError, sqlite3.Error, OSError):
            # A lease that could not be renewed this time is renewed by the
            # next call; the worker's idle limit is far longer than that.
            pass

    # Section: calls

    def call(
        self,
        target: Target,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        operation_id: str | None = None,
        scene_epoch: int | None = None,
        wait_s: float | None = None,
        timeout_s: float | None = None,
        socket_s: float | None = None,
        skip_if_busy: bool = False,
        cancelled: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Send one bridge call and hand back the reply, or raise `CallError`.

        `cancelled` is set when the caller has gone. A paced call still
        waiting for its turn then leaves the queue and is never sent.

        A call that carries an `operation_id` is sent once more if its reply is
        lost, with the same id, which the client does on its own. `socket_s`
        caps the wait on the socket, for a call that answers at once or not
        at all, such as a cancel. `skip_if_busy` asks for `SESSION_BUSY` at
        once rather than a place in the queue, including behind a main thread
        that is away.
        """
        # The same operation id as the call out is a resend: the bridge
        # answers it from that call's receipt without the main thread, so it
        # never waits for a turn behind the very call it asks about.
        resend = self.pacer.holds(target.session_id, operation_id)
        if resend or not self.paces(target, tool):
            return self._call(
                target,
                tool,
                arguments,
                operation_id=operation_id,
                scene_epoch=scene_epoch,
                wait_s=wait_s,
                timeout_s=timeout_s,
                socket_s=socket_s,
                skip_if_busy=skip_if_busy,
            )
        # The turn is waited for out of the caller's own queueing budget, and
        # the bridge is given what is left of it.
        budget = BRIDGE_WAIT_S if wait_s is None else wait_s
        try:
            turn = self.pacer.admit(
                target.session_id,
                budget_s=budget,
                skip_if_busy=skip_if_busy,
                cancelled=cancelled.is_set if cancelled is not None else None,
                operation_id=operation_id,
                runs_s=BRIDGE_TIMEOUT_S if timeout_s is None else timeout_s,
                runs_known=timeout_s is not None,
            )
        except NoTurn as refused:
            raise paced_busy(target, refused) from None
        if cancelled is not None and cancelled.is_set():
            # Gone in the moment between the turn and the send: give it back.
            self.pacer.done(target.session_id)
            raise paced_busy(
                target,
                NoTurn(waited_s=turn.waited_s, retry_after_s=0.0, reason="the caller went away"),
            )
        waited = throttled_ms(turn.waited_s)
        said: dict[str, Any] = {ADMITTED_KEY: round(turn.at, 3)}
        if waited:
            said[THROTTLED_KEY] = waited
            wait_s = max(0.0, budget - turn.waited_s)
        try:
            reply = self._call(
                target,
                tool,
                arguments,
                operation_id=operation_id,
                scene_epoch=scene_epoch,
                wait_s=wait_s,
                timeout_s=timeout_s,
                socket_s=socket_s,
                skip_if_busy=skip_if_busy,
            )
        except CallError as error:
            error.trace.update(said)
            raise
        finally:
            self.pacer.done(target.session_id)
        return {**reply, **said}

    def paces(self, target: Target, tool: str) -> bool:
        """Whether a call of this tool to this session waits its turn first."""
        if tool in UNPACED_TOOLS or not self.pacer.active:
            return False
        return target.record.kind == "gui" or (self.pace_workers and target.record.kind == "hython")

    def _call(
        self,
        target: Target,
        tool: str,
        arguments: Mapping[str, Any] | None,
        *,
        operation_id: str | None,
        scene_epoch: int | None,
        wait_s: float | None,
        timeout_s: float | None,
        socket_s: float | None,
        skip_if_busy: bool,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {"skip_if_busy": True} if skip_if_busy else {}
        try:
            answer = self._send(
                target.session,
                tool,
                arguments=dict(arguments or {}),
                session_id=target.session_id,
                scene_epoch=scene_epoch,
                operation_id=operation_id,
                wait_s=wait_s,
                timeout_s=timeout_s,
                http_timeout_s=socket_s if socket_s is not None else socket_wait(wait_s, timeout_s),
                **extra,
            )
        except client.BridgeUnreachable as error:
            self._lost(target, operation_id, error)
        except client.BridgeNotAuthentic as error:
            self._not_authentic(target, error)
        try:
            return self._reply(target, answer.payload)
        finally:
            # The session answered, so a worker stays warm from this moment,
            # not from when the call was sent.
            if target.record.kind == "hython":
                self._renew(target.session_id)

    def health(self, target: Target) -> dict[str, Any]:
        """What the session says about itself. Answers even while it is busy."""
        started = time.monotonic()
        try:
            answer = self._ask_health(target.session, timeout_s=HEALTH_TIMEOUT_S)
        except client.BridgeUnreachable as error:
            self._lost(target, None, error)
        except client.BridgeNotAuthentic as error:
            self._not_authentic(target, error)
        payload = self._reply(target, answer.payload)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise CallError("BAD_REPLY", "the health answer held no data")
        return {**data, "round_trip_ms": round((time.monotonic() - started) * 1000.0, 3)}

    def _reply(self, target: Target, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or "ok" not in payload:
            raise CallError(
                "BAD_REPLY",
                "the session answered with something that is not a reply",
                details={"session_id": target.session_id},
            )
        if payload["ok"]:
            return payload
        failure = CallError.from_reply(payload)
        if failure.code in STALE_CLIENT_CODES:
            self.forget(target.session_id)
        raise failure

    def _lost(self, target: Target, operation_id: str | None, error: Exception) -> NoReturn:
        """Nothing answered. Say whether the session has gone, or only went quiet."""
        self.forget(target.session_id)
        records = self._records()
        now = next((r for r in records if r.session_id == target.session_id), None)
        live = [r for r in records if r.state in LIVE_STATES]
        if now is None or now.state in DEAD_STATES:
            raise dead(target.session_id, target.record.alias, live) from error
        details: dict[str, Any] = {"session_id": target.session_id, "alias": target.record.alias}
        hint = None
        if operation_id:
            details["operation_id"] = operation_id
            hint = (
                "the change may have happened; send the same operation_id again"
                " to get its outcome rather than doing it twice"
            )
        raise CallError(
            "SESSION_UNREACHABLE",
            f"session {target.record.alias} did not answer",
            hint=hint,
            details=details,
        ) from error

    def _not_authentic(self, target: Target, error: Exception) -> NoReturn:
        self.forget(target.session_id)
        records = self._records()
        now = next((r for r in records if r.session_id == target.session_id), None)
        live = [r for r in records if r.state in LIVE_STATES]
        if now is None or now.state in DEAD_STATES:
            raise dead(target.session_id, target.record.alias, live) from error
        raise CallError(
            "REPLY_NOT_AUTHENTIC",
            f"the answer on {target.record.alias}'s port was not signed by it",
            details={"session_id": target.session_id, "alias": target.record.alias},
        ) from error


def paced_busy(target: Target, refused: NoTurn) -> CallError:
    """`SESSION_BUSY` for a call whose turn would come too late for it, or
    that asked not to wait. Nothing was sent, so nothing ran."""
    waited = throttled_ms(refused.waited_s)
    details: dict[str, Any] = {
        "session_id": target.session_id,
        "alias": target.record.alias,
        "paced": True,
        "reason": refused.reason,
        "retry_after_s": refused.retry_after_s,
        THROTTLED_KEY: waited,
    }
    return CallError(
        "SESSION_BUSY",
        f"this server paces its calls to {target.record.alias}, which has a user interface,"
        f" and its next turn is {refused.retry_after_s:g} seconds away; nothing was sent",
        hint=(
            "call again after retry_after_s, or pass a wait_s longer than that to queue"
            " for the turn"
        ),
        details=details,
        trace={THROTTLED_KEY: waited} if waited else None,
    )


def socket_wait(wait_s: float | None, timeout_s: float | None) -> float:
    """How long to wait on the socket: every budget the bridge may use, and a margin."""
    queued = BRIDGE_WAIT_S if wait_s is None else wait_s
    running = BRIDGE_TIMEOUT_S if timeout_s is None else timeout_s
    return queued + running + SOCKET_MARGIN_S


def session_facts(home: Path, session_id: str) -> dict[str, Any]:
    """The facts a session file holds beside its token, without the token."""
    try:
        entry = registry.read_entry(registry.entry_path(Path(home), session_id))
    except (OSError, ValueError):
        return {}
    entry.pop("token", None)
    return entry
