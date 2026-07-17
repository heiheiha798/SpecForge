# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Backpressure: the control plane's "when to pause rollout" decision.

Ownership (from ``control_plane/backpressure.md``): **the controller decides when
to pause; the FeatureStore only reports capacity.** This module is the policy. It
reads capacity through a narrow ``CapacityReporter`` (anything exposing
``health() -> dict`` — the feature store qualifies, and its health dict carries
*ints only*, never tensors, so the control plane stays tensor-free). It owns no
tensors and no scheduling state beyond its own counters.

Phase-1 policy (this file):

* pause prompt leasing when feature bytes cross a **high watermark**; resume only
  once they fall back below a **low watermark** (hysteresis, so we don't flap);
* pause at the producer's remote in-flight ref limit or while required shared
  artifact reclamation is pending;
* cap in-flight prompt tasks per rollout worker;
* cap sample refs leased to the trainer per call;
* count **rollout starvation** (paused, can't produce) and **trainer starvation**
  (queue empty when the trainer asks) *separately*, because they call for
  opposite fixes.

Later (not here): adaptive rollout batch size, priority routing for eval samples,
stale-sample dropping, per-strategy byte budgets, weighted prompt-source
scheduling.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


class CapacityReporter(Protocol):
    """Anything that can report data-plane capacity as a flat metadata dict."""

    def health(self) -> Dict[str, Any]: ...


@dataclass
class BackpressureConfig:
    """Watermarks and caps. All optional; ``None`` disables that lever.

    ``low_watermark_bytes`` must be ``<= high_watermark_bytes`` so the resume
    threshold sits below the pause threshold (the hysteresis band). If only the
    high watermark is given, the low defaults to it (degenerate: no hysteresis).
    """

    high_watermark_bytes: Optional[int] = None
    low_watermark_bytes: Optional[int] = None
    max_inflight_prompts_per_worker: Optional[int] = None
    max_train_lease: Optional[int] = None
    max_remote_in_flight: Optional[int] = None
    pause_on_release_pending: bool = False

    def __post_init__(self) -> None:
        if self.high_watermark_bytes is not None and self.low_watermark_bytes is None:
            self.low_watermark_bytes = self.high_watermark_bytes
        if (
            self.high_watermark_bytes is not None
            and self.low_watermark_bytes is not None
            and self.low_watermark_bytes > self.high_watermark_bytes
        ):
            raise ValueError(
                "low_watermark_bytes must be <= high_watermark_bytes "
                f"({self.low_watermark_bytes} > {self.high_watermark_bytes})"
            )
        if self.max_remote_in_flight is not None and self.max_remote_in_flight < 1:
            raise ValueError("max_remote_in_flight must be >= 1 or None")


class BackpressureController:
    """Pure policy: decides pause/resume and caps from capacity signals.

    Holds a latched ``_paused`` flag for hysteresis and separate starvation
    counters. Thread-safe so a rollout-lease thread and a trainer-lease thread
    can consult it concurrently.
    """

    def __init__(
        self,
        config: Optional[BackpressureConfig] = None,
        capacity: Optional[CapacityReporter] = None,
    ) -> None:
        self.config = config or BackpressureConfig()
        self.capacity = capacity
        self._paused = False
        self._byte_paused = False
        self._pause_reasons = ()
        self._lock = threading.Lock()
        self._stats = {
            "rollout_starved": 0,  # asked to lease prompts while paused
            "trainer_starved": 0,  # asked to lease train refs, queue was empty
            "pause_transitions": 0,
            "resume_transitions": 0,
        }

    # -- the pause decision (hysteresis) -----------------------------------
    def should_pause_prompts(self) -> bool:
        """True iff prompt leasing should be paused right now.

        Latches on at the high watermark and off at the low watermark so a store
        hovering near one threshold does not flap pause/resume every call.
        """
        health = self.capacity.health() if self.capacity is not None else {}
        hi = self.config.high_watermark_bytes
        resident = int(health.get("resident_bytes", 0))
        lo = self.config.low_watermark_bytes
        with self._lock:
            if hi is not None and not self._byte_paused and resident >= hi:
                self._byte_paused = True
            elif (
                hi is not None
                and lo is not None
                and self._byte_paused
                and resident <= lo
            ):
                self._byte_paused = False

            reasons = []
            if self._byte_paused:
                reasons.append("resident_bytes")
            remote_limit = self.config.max_remote_in_flight
            if (
                remote_limit is not None
                and int(health.get("remote_in_flight", 0)) >= remote_limit
            ):
                reasons.append("remote_in_flight")
            if self.config.pause_on_release_pending and (
                int(health.get("release_pending", 0)) > 0
                or int(health.get("required_reclaims_pending", 0)) > 0
            ):
                reasons.append("release_pending")

            paused = bool(reasons)
            if paused and not self._paused:
                self._stats["pause_transitions"] += 1
            elif self._paused and not paused:
                self._stats["resume_transitions"] += 1
            self._paused = paused
            self._pause_reasons = tuple(reasons)
            return self._paused

    # -- caps ---------------------------------------------------------------
    def cap_prompt_grant(self, worker_inflight: int, requested: int) -> int:
        """How many prompt tasks may be granted given the worker's in-flight count."""
        cap = self.config.max_inflight_prompts_per_worker
        if cap is None:
            return requested
        return max(0, min(requested, cap - worker_inflight))

    def cap_train_lease(self, requested: int) -> int:
        cap = self.config.max_train_lease
        return requested if cap is None else min(requested, cap)

    # -- starvation accounting ---------------------------------------------
    def note_rollout_starved(self) -> None:
        with self._lock:
            self._stats["rollout_starved"] += 1

    def note_trainer_starved(self) -> None:
        with self._lock:
            self._stats["trainer_starved"] += 1

    # -- observability ------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap: Dict[str, Any] = {
                "paused": self._paused,
                "pause_reasons": list(self._pause_reasons),
                **self._stats,
            }
        if self.capacity is not None:
            h = self.capacity.health()
            resident = int(h.get("resident_bytes", 0))
            snap["resident_bytes"] = resident
            snap["max_resident_bytes"] = h.get("max_resident_bytes")
            hi = self.config.high_watermark_bytes
            snap["free_to_high_watermark_bytes"] = (
                max(0, hi - resident) if hi is not None else None
            )
            snap["avg_feature_age_s"] = h.get("avg_age_s")
            snap["oldest_feature_age_s"] = h.get("oldest_age_s")
            snap["remote_in_flight"] = int(h.get("remote_in_flight", 0))
            snap["max_remote_in_flight"] = self.config.max_remote_in_flight
            snap["release_pending"] = int(h.get("release_pending", 0))
            snap["required_reclaims_pending"] = int(
                h.get("required_reclaims_pending", 0)
            )
        return snap


__all__ = ["CapacityReporter", "BackpressureConfig", "BackpressureController"]
