"""Receipts, so a repeated call does not do the work a second time.

A connection can close after Houdini has changed the scene and before the
answer arrives. The caller cannot tell that from a call that never ran, so a
blind retry builds the network twice, saves twice or renders twice. The answer
is a receipt taken before the work starts and finished with the outcome:

- A mutating call carries an `operation_id`. The id is bound to the session,
  the scene epoch and a digest of the tool name and its arguments.
- Claiming the id is what gives a caller the right to run the work. The claim
  and the row are one transaction in the coordination store, so two processes
  presenting the same id cannot both run it.
- The same id and the same arguments, once the first attempt has finished,
  returns the stored answer. Nothing runs again.
- The same id with different arguments is a mistake, not a retry, and is
  refused with `OPERATION_MISMATCH`.
- An id presented while another process is still working on it, or after an
  attempt that died without recording anything, gets `OUTCOME_UNKNOWN` with
  what the receipt does say. The work may have happened. Nothing here will
  pretend to know.
- An id presented against a scene that has been replaced is refused with
  `SCENE_REPLACED`, because the answer it would replay describes a scene that
  is gone.

Reads take no receipt. Repeating a read costs nothing and changes nothing.

The rows are the same rows the server side reads, so a receipt survives the
client process that made it. They are pruned by age, on a count of calls
rather than on a timer: a session nobody is calling has nothing to prune.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nscr_houdini_mcp import store as store_module

# How long a receipt can answer a retry. Long enough that a caller coming back
# after a crash and a restart still gets its answer.
DEFAULT_MAX_AGE_S = 24 * 60 * 60.0

# How many claims apart the pruning runs.
DEFAULT_PRUNE_EVERY = 50

RUN = "run"
REPLAY = "replay"
MISMATCH = "mismatch"
UNKNOWN = "unknown"
SCENE_REPLACED = "scene_replaced"
SKIP = "skip"

# Why an id is closed with no answer in it. The two cases are different and a
# caller is told which one it met.
ABANDONED_BY = "an earlier attempt stopped without recording an outcome"
STILL_RUNNING = "another caller is running this id"


@dataclass(frozen=True)
class Verdict:
    """What presenting an operation id came back with."""

    action: str
    outcome: Any = None
    record: store_module.OperationRecord | None = None
    recorded_epoch: int | None = None
    reason: str | None = None

    @property
    def may_run(self) -> bool:
        return self.action in (RUN, SKIP)

    def state(self) -> dict[str, Any]:
        """The receipt as a caller can be told about it, with no scene text."""
        record = self.record
        if record is None:
            return {}
        return {
            "operation_id": record.operation_id,
            "state": record.state,
            "session_id": record.session_id,
            "scene_epoch": record.scene_epoch,
            "job_id": record.job_id,
            "started_at": record.created_at,
            "updated_at": record.updated_at,
        }


def ended_epoch(record: store_module.OperationRecord) -> int | None:
    """The scene this operation left behind when it finished.

    The stored answer carries the epoch as of the moment it was given, which
    is what an operation that replaced the scene itself has to be judged
    against: its own load or new scene moved the counter, and that must not
    stop it answering its own retry.
    """
    outcome = record.outcome
    if not isinstance(outcome, Mapping):
        return None
    epoch = outcome.get("scene_epoch")
    return epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else None


def digest_call(tool: str, arguments: Mapping[str, Any]) -> str:
    """The digest an id is bound to: the tool name and the arguments together.

    The same id against a different tool is as much a mistake as the same id
    against different arguments, so both are in it.
    """
    return store_module.digest_arguments({"tool": tool, "arguments": dict(arguments)})


class Receipts:
    """The receipt table, as one call sees it.

    A store handle belongs to one thread, so this opens one per question and
    closes it again. The questions are small and rare next to the work they
    guard.
    """

    def __init__(
        self,
        store: Callable[[], store_module.Store] | None,
        *,
        session_id: str = "",
        owner_pid: int | None = None,
        log: Callable[[str], None] | None = None,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        prune_every: int = DEFAULT_PRUNE_EVERY,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._owner_pid = owner_pid
        self._log = log or (lambda text: None)
        self._max_age_s = max_age_s
        self._prune_every = max(1, prune_every)
        self._claims = 0

    @property
    def available(self) -> bool:
        """Whether there is anywhere to keep a receipt."""
        return self._store is not None

    def peek(self, operation_id: str, digest: str, *, current_epoch: int | None = None) -> Verdict:
        """Answer a repeat before it queues, without claiming anything.

        Only the answers that are certain without a transaction: a receipt
        with different arguments, and one that has already finished.
        Everything else waits for the claim, which is the one that decides.
        """
        record = self._read(lambda store: store.get_operation(operation_id))
        if record is None:
            return Verdict(RUN)
        if record.digest != digest:
            return Verdict(MISMATCH, record=record)
        return self._settled(record, current_epoch)

    def claim(
        self,
        operation_id: str,
        digest: str,
        *,
        scene_epoch: int | None = None,
        current_epoch: int | None = None,
    ) -> Verdict:
        """Take the id, or say what it already did.

        `scene_epoch` is the scene the caller wrote the call against, which is
        what the id is bound to and what a retry presents again.
        `current_epoch` is the scene there is now, which decides whether a
        stored answer still describes the scene a caller would get.
        """
        store = self._open()
        if store is None:
            return Verdict(SKIP, reason="no receipt store")
        try:
            claim = store.begin_operation(
                operation_id,
                digest,
                session_id=self._session_id or None,
                scene_epoch=scene_epoch,
                owner_pid=self._owner_pid,
            )
            if claim.claimed and not claim.outcome_unknown:
                return Verdict(RUN, record=claim.record)
            if claim.claimed:
                # An earlier attempt held this id and stopped without
                # recording anything. It may have changed the scene first, so
                # the id is closed rather than run again by anybody.
                return Verdict(
                    UNKNOWN, record=self._abandon(store, operation_id), reason=ABANDONED_BY
                )
            if claim.outcome_unknown:
                return Verdict(UNKNOWN, record=claim.record, reason=STILL_RUNNING)
            return self._settled(claim.record, current_epoch)
        except store_module.OperationMismatch:
            return Verdict(MISMATCH)
        except store_module.SceneReplaced as replaced:
            return Verdict(SCENE_REPLACED, recorded_epoch=replaced.recorded_epoch)
        except store_module.StoreError as error:
            self._log(f"could not claim {operation_id}: {type(error).__name__}: {error}")
            return Verdict(SKIP, reason=type(error).__name__)
        finally:
            self._prune(store)
            store.close()

    def _settled(self, record: store_module.OperationRecord, current_epoch: int | None) -> Verdict:
        """What a receipt this caller may not run says for itself."""
        if record.state == store_module.OPERATION_ABANDONED:
            return Verdict(UNKNOWN, record=record, reason=ABANDONED_BY)
        if record.state == "running":
            return Verdict(RUN, record=record)
        ended = ended_epoch(record)
        if current_epoch is None or ended is None or ended == current_epoch:
            return Verdict(REPLAY, outcome=record.outcome, record=record)
        # The scene has moved on since this answer was given, and not by this
        # operation, so replaying it would describe a scene that is gone.
        return Verdict(SCENE_REPLACED, record=record, recorded_epoch=ended)

    def _abandon(
        self, store: store_module.Store, operation_id: str
    ) -> store_module.OperationRecord | None:
        """Close a receipt whose attempt stopped without recording anything.

        It is a state of its own, so the next caller to present the id is told
        what really happened rather than that somebody else is working on it.
        """
        try:
            return store.finish_operation(
                operation_id,
                state=store_module.OPERATION_ABANDONED,
                error={"reason": ABANDONED_BY},
            )
        except store_module.StoreError as error:
            self._log(f"could not close {operation_id}: {type(error).__name__}: {error}")
            return None

    def finish(self, operation_id: str, payload: Mapping[str, Any]) -> None:
        """Store the answer, so a retry with the same id is answered from it."""
        done = bool(payload.get("ok"))
        self._write(
            lambda store: store.finish_operation(
                operation_id,
                state="done" if done else "failed",
                outcome=dict(payload),
                error=None if done else dict(payload.get("error") or {}),
            )
        )

    def drop(self, operation_id: str) -> None:
        """Take a receipt off a call that was refused before the tool ran.

        Nothing happened under that id, so the next caller presenting it gets
        a clean run rather than an outcome nobody knows.
        """
        self._write(lambda store: store.drop_operation(operation_id))

    def touch(self, operation_id: str) -> None:
        """Say the work is still going, so nobody else takes the receipt over."""
        self._write(lambda store: store.touch_operation(operation_id))

    # Section: the store handle

    def _open(self) -> store_module.Store | None:
        if self._store is None:
            return None
        try:
            return self._store()
        except Exception as error:  # noqa: BLE001 - a call still runs without receipts
            self._log(f"could not open the receipt store: {type(error).__name__}: {error}")
            return None

    def _read(self, ask: Callable[[store_module.Store], Any]) -> Any:
        store = self._open()
        if store is None:
            return None
        try:
            return ask(store)
        except store_module.StoreError as error:
            self._log(f"could not read a receipt: {type(error).__name__}: {error}")
            return None
        finally:
            store.close()

    def _write(self, change: Callable[[store_module.Store], Any]) -> None:
        store = self._open()
        if store is None:
            return
        try:
            change(store)
        except store_module.StoreError as error:
            self._log(f"could not write a receipt: {type(error).__name__}: {error}")
        finally:
            store.close()

    def _prune(self, store: store_module.Store) -> None:
        """Drop receipts past their retention, every so many claims."""
        self._claims += 1
        if self._claims % self._prune_every:
            return
        try:
            dropped = store.prune_operations(self._max_age_s)
        except store_module.StoreError as error:
            self._log(f"could not prune receipts: {type(error).__name__}: {error}")
            return
        if dropped:
            self._log(f"dropped {dropped} receipts older than {self._max_age_s:g} seconds")
