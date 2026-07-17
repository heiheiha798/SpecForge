# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fixed-subscription fan-out for independent online consumers.

``RefDistributor`` shards refs across ranks of one lockstep trainer. This module
instead gives every logical consumer the same ordered capture stream. Source
backpressure advances by the minimum consumer cursor, so a shared artifact is
reclaimable only after every consumer has acknowledged it.

The subscription set is immutable. Resume is explicit: callers provide each
consumer's optimizer-durable prefix cursor. The distributor rewinds the source
counter to the minimum cursor and rebuilds only each consumer's uncommitted
suffix. Materialization-only ACKs from a crashed attempt are never trusted. A
restarted producer must also call ``StreamingRefChannel.seed_published()`` before
using its in-flight backpressure count.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import threading
import time
import traceback
from collections import deque
from typing import Callable, Deque, Dict, Iterable, Mapping, Optional, Tuple

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

logger = logging.getLogger(__name__)

_FAILED_SUFFIX = ".failed"
_CONSUMER_FAILED_SUFFIX = ".consumer_failed"
_OWNED_SUFFIXES = (
    "",
    ".closed",
    ".consumed_count",
    ".consumed_count.tmp",
    _FAILED_SUFFIX,
    _FAILED_SUFFIX + ".tmp",
    _CONSUMER_FAILED_SUFFIX,
    _CONSUMER_FAILED_SUFFIX + ".tmp",
)
_CONSUMER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _atomic_text(path: str, value: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def durable_prefix_cursor(
    refs: Iterable[SampleRef], acked_sample_ids: Iterable[str]
) -> int:
    """Validate durable ACK identity and return its ordered prefix cursor.

    A cursor is safe for replay only when the ACK set is exactly the first N
    unique samples in source order. Holes, unknown IDs, and duplicate source IDs
    are rejected instead of silently skipping or retraining captures.
    """
    ordered = [ref.sample_id for ref in refs]
    if len(set(ordered)) != len(ordered):
        raise ValueError("source history contains duplicate sample IDs")
    acked = set(acked_sample_ids)
    unknown = acked - set(ordered)
    if unknown:
        raise ValueError(
            f"durable ACKs contain IDs absent from source: {sorted(unknown)}"
        )
    cursor = len(acked)
    expected = set(ordered[:cursor])
    if acked != expected:
        missing = expected - acked
        out_of_order = acked - expected
        raise ValueError(
            "durable ACKs are not an exact source prefix "
            f"(missing={sorted(missing)}, out_of_order={sorted(out_of_order)})"
        )
    return cursor


class BroadcastSubscriptionChannel(StreamingRefChannel):
    """Failure-aware channel for one stable logical consumer."""

    def _raise_if_failed(self) -> None:
        try:
            with open(self.path + _FAILED_SUFFIX) as f:
                detail = f.read()
        except FileNotFoundError:
            return
        raise RuntimeError(
            "broadcast-ref-distributor failed" + (f":\n{detail}" if detail else "")
        )

    def poll(self, max_n: Optional[int] = None) -> list[SampleRef]:
        self._raise_if_failed()
        refs = super().poll(max_n=max_n)
        self._raise_if_failed()
        return refs

    def is_closed(self) -> bool:
        self._raise_if_failed()
        return super().is_closed()

    _is_closed = is_closed

    def report_failure(self, error: BaseException | str) -> None:
        """Report terminal consumer failure to the fan-out authority."""
        if isinstance(error, BaseException):
            detail = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
        else:
            detail = str(error)
        _atomic_text(self.path + _CONSUMER_FAILED_SUFFIX, detail)


class BroadcastRefDistributor:
    """Broadcast one append-only source to fixed independent consumers.

    ``finished`` is exact: source EOF has propagated, all consumers acknowledged
    the full stream, and every global-prefix callback completed. Any partial
    publish, source error, consumer failure, cleanup error, timeout, or early stop
    is terminal failure and poisons every subscription.
    """

    def __init__(
        self,
        source: StreamingRefChannel,
        subscription_root: str,
        consumer_ids: Iterable[str],
        *,
        resume_cursors: Optional[Mapping[str, int]] = None,
        max_inflight_refs: Optional[int] = None,
        on_global_consumed: Optional[Callable[[Tuple[SampleRef, ...]], None]] = None,
        on_abandoned: Optional[
            Callable[[Tuple[SampleRef, ...], BaseException], None]
        ] = None,
        poll_s: float = 0.05,
        idle_timeout_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        channel_factory: Callable[[str], StreamingRefChannel] = StreamingRefChannel,
    ) -> None:
        ids = self._validate_consumer_ids(consumer_ids)
        if poll_s < 0:
            raise ValueError(f"poll_s must be >= 0, got {poll_s}")
        if idle_timeout_s is not None and idle_timeout_s <= 0:
            raise ValueError(
                f"idle_timeout_s must be > 0 or None, got {idle_timeout_s}"
            )
        if max_inflight_refs is not None and (
            isinstance(max_inflight_refs, bool)
            or not isinstance(max_inflight_refs, int)
            or max_inflight_refs < 1
        ):
            raise ValueError(
                "max_inflight_refs must be a positive int or None, got "
                f"{max_inflight_refs!r}"
            )
        for name, callback in (
            ("on_global_consumed", on_global_consumed),
            ("on_abandoned", on_abandoned),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None")
        if not callable(channel_factory):
            raise TypeError("channel_factory must be callable")

        self.source = source
        self.subscription_root = os.path.abspath(subscription_root)
        self.consumer_ids = ids
        self.max_inflight_refs = max_inflight_refs
        self.on_global_consumed = on_global_consumed
        self.on_abandoned = on_abandoned
        self.poll_s = poll_s
        self.idle_timeout_s = idle_timeout_s
        self._clock = clock
        self._sleep = sleep

        paths = {
            consumer_id: self.subscription_path(self.subscription_root, consumer_id)
            for consumer_id in ids
        }
        source_artifacts = {
            os.path.abspath(source.path) + suffix for suffix in _OWNED_SUFFIXES
        }
        output_artifacts = {
            path + suffix for path in paths.values() for suffix in _OWNED_SUFFIXES
        }
        collisions = sorted(source_artifacts & output_artifacts)
        if collisions:
            raise ValueError(
                "source channel artifacts collide with broadcast subscription "
                f"artifacts: {collisions}"
            )

        cursors = self._validate_resume_cursors(ids, resume_cursors)
        resumed = resume_cursors is not None
        baseline = min(cursors.values())
        if resumed and max(cursors.values()) > 0:
            history = StreamingRefChannel(source.path).poll()
            if max(cursors.values()) > len(history):
                raise ValueError(
                    "resume cursor exceeds the complete source history available "
                    f"at construction ({max(cursors.values())} > {len(history)})"
                )
        if resumed:
            # Durable training cursors are authoritative. Rewind any ACKs from
            # materialized-but-not-durable batches before replay starts.
            self.source.set_consumed(baseline)
        else:
            source_consumed = self.source.seed_consumed()
            if source_consumed:
                raise ValueError(
                    "broadcast fan-out found prior source progress; provide "
                    "resume_cursors from each consumer's durable ledger or use a "
                    f"fresh capture run (consumed_count={source_consumed})"
                )

        # Subscription files are replay products, never recovery authority.
        os.makedirs(self.subscription_root, exist_ok=True)
        for path in paths.values():
            for suffix in _OWNED_SUFFIXES:
                try:
                    os.remove(path + suffix)
                except FileNotFoundError:
                    pass

        self._subscriptions = {
            consumer_id: channel_factory(paths[consumer_id]) for consumer_id in ids
        }
        self._paths = paths
        self._resume_cursors = cursors
        self._resumed = resumed
        self._resumed_global = baseline

        self._pending: Deque[SampleRef] = deque()
        self._abandoned: Deque[SampleRef] = deque()
        self._consumed = dict(cursors)
        self._broadcasted = 0
        self._global_consumed = baseline
        self._released = 0
        self._source_forwarded = 0
        self._source_closed = False
        self._outputs_closed = False
        self._partial_publish_failures = 0
        self._cleanup_error: Optional[BaseException] = None
        self._failure_cleanup_done = False

        self._stop = threading.Event()
        self._done = threading.Event()
        self._pump_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self.finished = False
        self.error: Optional[BaseException] = None

    @classmethod
    def with_feature_store(
        cls,
        source: StreamingRefChannel,
        subscription_root: str,
        consumer_ids: Iterable[str],
        *,
        feature_store,
        **kwargs,
    ) -> "BroadcastRefDistributor":
        """Build a distributor whose owner reclaims shared artifacts exactly once."""
        if "on_global_consumed" in kwargs or "on_abandoned" in kwargs:
            raise TypeError(
                "with_feature_store owns the lifecycle callbacks; construct "
                "BroadcastRefDistributor directly to supply custom callbacks"
            )
        reclaim = getattr(feature_store, "reclaim", None)
        if not callable(reclaim):
            raise TypeError("feature_store must expose reclaim(sample_ref, reason=...)")

        def reclaim_consumed(refs: Tuple[SampleRef, ...]) -> None:
            for ref in refs:
                reclaim(ref, reason="fanout-globally-consumed")

        def reclaim_abandoned(
            refs: Tuple[SampleRef, ...], error: BaseException
        ) -> None:
            reason = f"fanout-abandoned:{type(error).__name__}"
            for ref in refs:
                reclaim(ref, reason=reason)

        return cls(
            source,
            subscription_root,
            consumer_ids,
            on_global_consumed=reclaim_consumed,
            on_abandoned=reclaim_abandoned,
            **kwargs,
        )

    @staticmethod
    def _validate_consumer_ids(consumer_ids: Iterable[str]) -> Tuple[str, ...]:
        if isinstance(consumer_ids, (str, bytes)):
            raise ValueError("consumer_ids must be an iterable of IDs, not a string")
        ids = tuple(consumer_ids)
        if not ids:
            raise ValueError("at least one consumer_id is required")
        for consumer_id in ids:
            if not isinstance(consumer_id, str) or not _CONSUMER_ID.fullmatch(
                consumer_id
            ):
                raise ValueError(
                    f"invalid consumer_id {consumer_id!r}; use 1-128 ASCII letters, "
                    "digits, '.', '_' or '-', starting with a letter or digit"
                )
        duplicates = sorted(
            {consumer_id for consumer_id in ids if ids.count(consumer_id) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate consumer_ids: {duplicates}")
        return ids

    @staticmethod
    def _validate_resume_cursors(
        consumer_ids: Tuple[str, ...], cursors: Optional[Mapping[str, int]]
    ) -> Dict[str, int]:
        if cursors is None:
            return {consumer_id: 0 for consumer_id in consumer_ids}
        if not isinstance(cursors, Mapping):
            raise TypeError("resume_cursors must be a mapping or None")
        expected, actual = set(consumer_ids), set(cursors)
        if actual != expected:
            raise ValueError(
                "resume_cursors keys must exactly match consumer_ids "
                f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
            )
        out: Dict[str, int] = {}
        for consumer_id in consumer_ids:
            cursor = cursors[consumer_id]
            if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
                raise ValueError(
                    f"resume cursor for {consumer_id!r} must be a non-negative int, "
                    f"got {cursor!r}"
                )
            out[consumer_id] = cursor
        return out

    @staticmethod
    def subscription_path(subscription_root: str, consumer_id: str) -> str:
        ids = BroadcastRefDistributor._validate_consumer_ids([consumer_id])
        return os.path.join(os.path.abspath(subscription_root), f"{ids[0]}.jsonl")

    @property
    def subscription_paths(self) -> Dict[str, str]:
        return dict(self._paths)

    def _ensure_active(self) -> None:
        if self.error is not None:
            raise RuntimeError(
                "broadcast-ref-distributor is already failed"
            ) from self.error
        if self._stop.is_set() and not self.finished:
            raise RuntimeError("broadcast-ref-distributor stopped before completion")

    def _check_consumer_failures(self) -> None:
        for consumer_id, path in self._paths.items():
            try:
                with open(path + _CONSUMER_FAILED_SUFFIX) as f:
                    detail = f.read().strip()
            except FileNotFoundError:
                continue
            raise RuntimeError(
                f"logical consumer {consumer_id!r} reported terminal failure"
                + (f": {detail}" if detail else "")
            )

    def _publish(self, ref: SampleRef) -> None:
        ordinal = self._broadcasted
        targets = [
            consumer_id
            for consumer_id in self.consumer_ids
            if ordinal >= self._resume_cursors[consumer_id]
        ]
        published = 0
        try:
            for consumer_id in targets:
                self._ensure_active()
                self._subscriptions[consumer_id].publish(ref)
                published += 1
        except BaseException:
            with self._state_lock:
                self._abandoned.append(ref)
                if published:
                    self._partial_publish_failures += 1
            raise
        with self._state_lock:
            self._broadcasted += 1
            if self._broadcasted > self._global_consumed:
                self._pending.append(ref)

    def _refresh_consumed(self) -> bool:
        previous_global = self._global_consumed
        observed: Dict[str, int] = {}
        progress = False
        for consumer_id in self.consumer_ids:
            remote = self._subscriptions[consumer_id].consumed_remote()
            delivered = max(0, self._broadcasted - self._resume_cursors[consumer_id])
            if remote > delivered:
                raise RuntimeError(
                    f"consumer {consumer_id!r} acknowledged {remote} replay refs, "
                    f"but only {delivered} were delivered"
                )
            count = self._resume_cursors[consumer_id] + remote
            previous = self._consumed[consumer_id]
            if count < previous:
                raise RuntimeError(
                    f"consumer {consumer_id!r} cursor regressed from {previous} to {count}"
                )
            progress = progress or count > previous
            observed[consumer_id] = count

        with self._state_lock:
            self._ensure_active()
            self._consumed.update(observed)
            new_global = min(observed.values())
            if new_global == previous_global:
                return progress
            if new_global < previous_global or new_global > self._broadcasted:
                raise RuntimeError(
                    f"invalid global consumer cursor transition "
                    f"{previous_global}->{new_global} at broadcasted={self._broadcasted}"
                )
            delta = new_global - previous_global
            if delta > len(self._pending):
                raise RuntimeError(
                    f"global consumed delta {delta} exceeds pending refs "
                    f"{len(self._pending)}"
                )
            self._global_consumed = new_global
            refs = tuple(itertools.islice(self._pending, 0, delta))

        if self.on_global_consumed is not None:
            self.on_global_consumed(refs)
        # Source progress is published only after shared-artifact reclaim succeeds.
        self.source.mark_consumed(delta)
        with self._state_lock:
            self._ensure_active()
            for _ in range(delta):
                self._pending.popleft()
            self._released += delta
            self._source_forwarded += delta
        return True

    def _remaining_capacity(self) -> Optional[int]:
        if self.max_inflight_refs is None:
            return None
        with self._state_lock:
            # During resume, replay of the already-global prefix consumes no
            # retention capacity. New refs stop exactly at global + the hard cap.
            return max(
                0,
                self._global_consumed + self.max_inflight_refs - self._broadcasted,
            )

    def _close_outputs(self) -> None:
        invalid = {
            consumer_id: cursor
            for consumer_id, cursor in self._resume_cursors.items()
            if cursor > self._broadcasted
        }
        if invalid:
            raise RuntimeError(
                f"resume cursors exceed source history ({self._broadcasted} refs): {invalid}"
            )
        for consumer_id in self.consumer_ids:
            self._ensure_active()
            self._subscriptions[consumer_id].close()
        with self._state_lock:
            self._outputs_closed = True
            self._source_closed = True

    def _finish_if_drained(self) -> bool:
        with self._state_lock:
            if not self._source_closed or self._global_consumed != self._broadcasted:
                return False
            if self._pending:
                raise RuntimeError("global drain reached with pending broadcast refs")
            self._ensure_active()
            self.finished = True
        logger.info("broadcast-ref-distributor: finished %s", self.stats())
        self._done.set()
        return True

    def _pump_once(self) -> bool:
        self._ensure_active()
        if self.finished:
            return False
        self._check_consumer_failures()
        progress = self._refresh_consumed()

        if not self._source_closed:
            capacity = self._remaining_capacity()
            if capacity is None or capacity > 0:
                raw = self.source.poll(max_n=capacity)
                if raw:
                    for index, ref in enumerate(raw):
                        try:
                            self._publish(ref)
                        except BaseException:
                            with self._state_lock:
                                self._abandoned.extend(raw[index + 1 :])
                            raise
                    progress = True
                elif self.source.is_closed():
                    self._close_outputs()
                    progress = True

        if self._refresh_consumed():
            progress = True
        if self._finish_if_drained():
            progress = True
        return progress

    def pump(self) -> bool:
        """Run one non-blocking fan-out, ACK, and terminal-state cycle."""
        with self._pump_lock:
            try:
                return self._pump_once()
            except BaseException as exc:  # noqa: BLE001 - terminal state is explicit
                self._record_failure(exc)
                raise

    @staticmethod
    def _unique_refs(refs: Iterable[SampleRef]) -> Tuple[SampleRef, ...]:
        out = []
        seen = set()
        for ref in refs:
            identity = (ref.sample_id, ref.feature_store_uri)
            if identity in seen:
                continue
            seen.add(identity)
            out.append(ref)
        return tuple(out)

    def _poison_subscriptions(self, error: BaseException) -> None:
        detail = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        if self._cleanup_error is not None:
            detail += "\nAbandoned cleanup also failed:\n" + "".join(
                traceback.format_exception(
                    type(self._cleanup_error),
                    self._cleanup_error,
                    self._cleanup_error.__traceback__,
                )
            )
        for path in self._paths.values():
            try:
                _atomic_text(path + _FAILED_SUFFIX, detail)
            except OSError:
                logger.exception("could not poison broadcast subscription %s", path)

    def _record_failure(self, error: BaseException) -> None:
        try:
            source_snapshot = StreamingRefChannel(self.source.path).poll()
        except BaseException:  # noqa: BLE001 - best-effort cleanup inventory
            source_snapshot = []
            logger.exception("could not inventory source refs during fan-out failure")
        with self._state_lock:
            if self.error is not None or self.finished:
                return
            self.error = error
            abandoned = self._unique_refs(
                (*source_snapshot, *self._pending, *self._abandoned)
            )
            run_cleanup = not self._failure_cleanup_done
            self._failure_cleanup_done = True
        if run_cleanup and self.on_abandoned is not None and abandoned:
            try:
                self.on_abandoned(abandoned, error)
            except BaseException as cleanup_error:  # noqa: BLE001 - reported in health
                self._cleanup_error = cleanup_error
                logger.exception("broadcast abandoned-artifact cleanup failed")
        logger.error(
            "broadcast-ref-distributor failed; poisoning subscriptions",
            exc_info=(type(error), error, error.__traceback__),
        )
        self._poison_subscriptions(error)
        self._done.set()

    def run(self) -> None:
        """Run until exact success or terminal failure."""
        last_progress = self._clock()
        while not self.finished:
            progress = self.pump()
            now = self._clock()
            if progress:
                last_progress = now
                continue
            if (
                self.idle_timeout_s is not None
                and now - last_progress > self.idle_timeout_s
            ):
                error = TimeoutError(
                    "broadcast-ref-distributor: no source or consumer ACK progress for "
                    f"{self.idle_timeout_s:.1f}s"
                )
                self._record_failure(error)
                raise error
            self._sleep(self.poll_s)

    def _run_guarded(self) -> None:
        try:
            self.run()
        except BaseException as exc:  # noqa: BLE001 - exposed through wait/health
            self._record_failure(exc)

    def start(self) -> "BroadcastRefDistributor":
        with self._state_lock:
            if self._started:
                if self._thread is not None and self._thread.is_alive():
                    return self
                raise RuntimeError("broadcast-ref-distributor cannot be restarted")
            if self.error is not None or self._stop.is_set():
                raise RuntimeError("cannot start a stopped or failed distributor")
            self._started = True
            self._thread = threading.Thread(
                target=self._run_guarded,
                name="broadcast-ref-distributor",
                daemon=True,
            )
            self._thread.start()
        return self

    def wait(
        self, timeout_s: Optional[float] = None, *, raise_on_error: bool = True
    ) -> bool:
        """Wait for exact success or failure; return ``False`` only on timeout."""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError(f"timeout_s must be >= 0 or None, got {timeout_s}")
        started = time.monotonic()
        if not self._done.wait(timeout=timeout_s):
            return False
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            remaining = None
            if timeout_s is not None:
                remaining = max(0.0, timeout_s - (time.monotonic() - started))
            thread.join(timeout=remaining)
            if thread.is_alive():
                return False
        if raise_on_error and self.error is not None:
            raise RuntimeError("broadcast-ref-distributor failed") from self.error
        return True

    join = wait

    def fail(self, error: BaseException, join_timeout_s: float = 10.0) -> None:
        """Force terminal failure from an external producer or supervisor."""
        if not isinstance(error, BaseException):
            raise TypeError(f"error must be an exception, got {type(error).__name__}")
        if join_timeout_s < 0:
            raise ValueError(f"join_timeout_s must be >= 0, got {join_timeout_s}")
        self._record_failure(error)
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=join_timeout_s)
            if thread.is_alive() and self._cleanup_error is None:
                self._cleanup_error = TimeoutError(
                    "broadcast-ref-distributor thread did not stop after failure"
                )

    def stop(self, join_timeout_s: float = 10.0) -> None:
        """Stop idempotently; stopping before drain is terminal failure."""
        if join_timeout_s < 0:
            raise ValueError(f"join_timeout_s must be >= 0, got {join_timeout_s}")
        with self._state_lock:
            terminal = self.finished or self.error is not None
            if not terminal:
                self._stop.set()
        thread = self._thread
        if terminal:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=join_timeout_s)
            return
        if thread is None or not thread.is_alive():
            self._record_failure(
                RuntimeError("broadcast-ref-distributor stopped before completion")
            )
            return
        if thread is not threading.current_thread():
            thread.join(timeout=join_timeout_s)
        if thread.is_alive():
            self._record_failure(
                TimeoutError(
                    "broadcast-ref-distributor thread did not stop within "
                    f"{join_timeout_s:.1f}s"
                )
            )

    def stats(self) -> Dict[str, object]:
        with self._state_lock:
            consumed = dict(self._consumed)
            delivered = {
                consumer_id: max(
                    0, self._broadcasted - self._resume_cursors[consumer_id]
                )
                for consumer_id in self.consumer_ids
            }
            lag = {
                consumer_id: max(0, self._broadcasted - count)
                for consumer_id, count in consumed.items()
            }
            return {
                "broadcasted": self._broadcasted,
                "global_consumed": self._global_consumed,
                "released": self._released,
                "source_forwarded": self._source_forwarded,
                "pending": len(self._pending),
                "consumed": consumed,
                "delivered": delivered,
                "lag": lag,
                "resume_cursors": dict(self._resume_cursors),
                "resumed_global": self._resumed_global,
                "max_inflight_refs": self.max_inflight_refs,
                "partial_publish_failures": self._partial_publish_failures,
            }

    def health(self) -> Dict[str, object]:
        with self._state_lock:
            if self.error is not None:
                state = "failed"
            elif self.finished:
                state = "finished"
            elif self._source_closed:
                state = "draining"
            elif self._started:
                state = "running"
            else:
                state = "initialized"
            return {
                "state": state,
                "finished": self.finished,
                "resumed": self._resumed,
                "source_closed": self._source_closed,
                "outputs_closed": self._outputs_closed,
                "thread_alive": self._thread is not None and self._thread.is_alive(),
                "error": None if self.error is None else repr(self.error),
                "cleanup_error": (
                    None if self._cleanup_error is None else repr(self._cleanup_error)
                ),
                **self.stats(),
            }


__all__ = [
    "BroadcastRefDistributor",
    "BroadcastSubscriptionChannel",
    "durable_prefix_cursor",
]
