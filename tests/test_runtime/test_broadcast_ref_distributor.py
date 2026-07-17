# coding=utf-8
"""Deterministic fixed-subscription fan-out lifecycle tests."""

import os
import tempfile
import time
import unittest

import torch

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.broadcast_ref_distributor import (
    BroadcastRefDistributor,
    BroadcastSubscriptionChannel,
    durable_prefix_cursor,
)
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import (
    StreamingRefChannel,
    StreamingRefQueue,
)


def _ref(index: int) -> SampleRef:
    sample_id = f"capture:s{index}"
    return SampleRef(
        sample_id=sample_id,
        run_id="capture",
        source_task_id=f"task-{index}",
        feature_store_uri=f"mooncake://capture/{sample_id}",
        feature_keys={"hidden_state": f"{sample_id}/hidden_state"},
        feature_specs={},
        strategy="dflash",
        metadata={"generation": 1},
    )


class _FailingSource(StreamingRefChannel):
    def poll(self, max_n=None):
        raise OSError("injected producer failure")


class _FailingPublishChannel(StreamingRefChannel):
    def publish(self, ref):
        raise OSError("injected subscription append failure")


class TestBroadcastRefDistributor(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="fanout_core_")
        self.source_path = os.path.join(self.root, "source.jsonl")
        self.outputs = os.path.join(self.root, "subscriptions")
        self.producer = StreamingRefChannel(self.source_path)

    def _distributor(self, consumers=("trainer-a",), **kwargs):
        return BroadcastRefDistributor(
            StreamingRefChannel(self.source_path),
            self.outputs,
            consumers,
            poll_s=0.0,
            **kwargs,
        )

    @staticmethod
    def _queue(path):
        return StreamingRefQueue(BroadcastSubscriptionChannel(path), poll_s=0.0)

    def _publish(self, count, *, close=True):
        refs = [_ref(i) for i in range(count)]
        self.producer.publish_many(refs)
        if close:
            self.producer.close()
        return refs

    def test_one_consumer_preserves_default_order_and_exact_terminal(self):
        refs = self._publish(4)
        reclaimed = []
        dist = self._distributor(
            on_global_consumed=lambda batch: reclaimed.extend(batch)
        )
        self.assertTrue(dist.pump())  # broadcast
        self.assertTrue(dist.pump())  # propagate source EOF

        queue = self._queue(dist.subscription_paths["trainer-a"])
        got = queue.get(4)
        self.assertEqual(
            [ref.sample_id for ref in got], [ref.sample_id for ref in refs]
        )
        queue.ack(got)
        queue.ack(got)  # duplicate ACK is an idempotent terminal effect

        self.assertTrue(dist.pump())
        self.assertTrue(dist.finished)
        self.assertEqual(self.producer.consumed_remote(), 4)
        self.assertEqual(
            [ref.sample_id for ref in reclaimed], [ref.sample_id for ref in refs]
        )
        self.assertEqual(queue.get(1), [])

    def test_three_consumers_receive_once_and_reclaim_at_minimum_ack(self):
        refs = self._publish(5)
        reclaimed = []
        dist = self._distributor(
            consumers=("fast", "medium", "slow"),
            on_global_consumed=lambda batch: reclaimed.append(
                [ref.sample_id for ref in batch]
            ),
        )
        dist.pump()
        dist.pump()
        queues = {
            consumer: self._queue(path)
            for consumer, path in dist.subscription_paths.items()
        }
        received = {consumer: queue.get(5) for consumer, queue in queues.items()}
        expected = [ref.sample_id for ref in refs]
        for batch in received.values():
            self.assertEqual([ref.sample_id for ref in batch], expected)

        queues["fast"].ack(received["fast"])
        queues["medium"].ack(received["medium"][:4])
        dist.pump()
        self.assertEqual(self.producer.consumed_remote(), 0)

        queues["slow"].ack(received["slow"][:2])
        dist.pump()
        self.assertEqual(self.producer.consumed_remote(), 2)
        self.assertEqual(reclaimed, [expected[:2]])

        queues["slow"].ack(received["slow"][2:])
        dist.pump()
        self.assertEqual(self.producer.consumed_remote(), 4)
        queues["medium"].ack(received["medium"][4:])
        dist.pump()
        self.assertTrue(dist.finished)
        self.assertEqual(reclaimed, [expected[:2], expected[2:4], expected[4:]])

    def test_feature_store_owner_retains_one_capture_until_all_consumers_ack(self):
        store = LocalFeatureStore("capture", retain_on_release=True)
        refs = [
            store.put(
                {"hidden_state": torch.tensor([index])},
                sample_id=f"capture:s{index}",
                metadata={"run_id": "capture", "strategy": "dflash"},
            )
            for index in range(3)
        ]
        self.producer.publish_many(refs)
        self.producer.close()
        dist = BroadcastRefDistributor.with_feature_store(
            StreamingRefChannel(self.source_path),
            self.outputs,
            ("trainer-a", "trainer-b", "trainer-c"),
            feature_store=store,
            poll_s=0.0,
        )
        dist.pump()
        dist.pump()
        queues = [self._queue(path) for path in dist.subscription_paths.values()]
        batches = [queue.get(3) for queue in queues]
        for batch in batches:
            for ref in batch:
                _, handle = store.get(ref)
                store.release(handle)
        queues[0].ack(batches[0])
        queues[1].ack(batches[1])
        dist.pump()
        self.assertEqual(store.health()["resident_samples"], 3)
        queues[2].ack(batches[2])
        dist.pump()
        self.assertTrue(dist.finished)
        self.assertEqual(store.health()["resident_samples"], 0)

    def test_external_failure_reclaims_broadcast_and_unread_store_artifacts(self):
        store = LocalFeatureStore("capture", retain_on_release=True)
        refs = [
            store.put(
                {"hidden_state": torch.tensor([index])},
                sample_id=f"capture:s{index}",
                metadata={"run_id": "capture", "strategy": "dflash"},
            )
            for index in range(4)
        ]
        self.producer.publish_many(refs)
        dist = BroadcastRefDistributor.with_feature_store(
            StreamingRefChannel(self.source_path),
            self.outputs,
            ("trainer-a", "trainer-b"),
            feature_store=store,
            max_inflight_refs=1,
            poll_s=0.0,
        )
        dist.pump()
        dist.fail(RuntimeError("producer worker failed"))
        self.assertEqual(dist.health()["state"], "failed")
        self.assertEqual(store.health()["resident_samples"], 0)

    def test_hard_inflight_cap_blocks_broadcast_until_slowest_ack(self):
        self._publish(10, close=False)
        dist = self._distributor(consumers=("fast", "slow"), max_inflight_refs=3)
        dist.pump()
        self.assertEqual(dist.stats()["broadcasted"], 3)
        self.assertEqual(dist.stats()["pending"], 3)

        fast = self._queue(dist.subscription_paths["fast"])
        slow = self._queue(dist.subscription_paths["slow"])
        fast_refs, slow_refs = fast.get(3), slow.get(3)
        fast.ack(fast_refs)
        dist.pump()
        self.assertEqual(dist.stats()["broadcasted"], 3)
        self.assertEqual(self.producer.consumed_remote(), 0)

        slow.ack(slow_refs[:1])
        dist.pump()
        self.assertEqual(self.producer.consumed_remote(), 1)
        self.assertEqual(dist.stats()["broadcasted"], 4)
        self.assertEqual(dist.stats()["pending"], 3)

    def test_resume_rebuilds_only_each_consumers_uncommitted_suffix(self):
        refs = self._publish(6)
        # The crashed attempt materialized farther than its durable checkpoints.
        self.producer.set_consumed(5)
        reclaimed = []
        dist = self._distributor(
            consumers=("trainer-a", "trainer-b"),
            resume_cursors={"trainer-a": 4, "trainer-b": 2},
            on_global_consumed=lambda batch: reclaimed.extend(batch),
        )
        self.assertEqual(self.producer.consumed_remote(), 2)
        restarted_producer = StreamingRefChannel(self.source_path)
        self.assertEqual(restarted_producer.seed_published(), 6)
        self.assertEqual(restarted_producer.in_flight_remote(), 4)
        dist.pump()
        dist.pump()

        queue_a = self._queue(dist.subscription_paths["trainer-a"])
        queue_b = self._queue(dist.subscription_paths["trainer-b"])
        got_a, got_b = queue_a.get(2), queue_b.get(4)
        self.assertEqual(
            [ref.sample_id for ref in got_a], [ref.sample_id for ref in refs[4:]]
        )
        self.assertEqual(
            [ref.sample_id for ref in got_b], [ref.sample_id for ref in refs[2:]]
        )

        queue_a.ack(got_a)
        queue_b.ack(got_b[:2])
        dist.pump()
        self.assertEqual(self.producer.consumed_remote(), 4)
        self.assertEqual(
            [ref.sample_id for ref in reclaimed], [ref.sample_id for ref in refs[2:4]]
        )
        queue_b.ack(got_b[2:])
        dist.pump()
        self.assertTrue(dist.finished)
        self.assertEqual(self.producer.consumed_remote(), 6)
        self.assertEqual(
            [ref.sample_id for ref in reclaimed], [ref.sample_id for ref in refs[2:]]
        )

    def test_fresh_run_rejects_prior_progress_without_touching_outputs(self):
        self.producer.set_consumed(1)
        os.makedirs(self.outputs)
        path = BroadcastRefDistributor.subscription_path(self.outputs, "trainer-a")
        with open(path, "w") as f:
            f.write("preserve-on-rejection")
        with self.assertRaisesRegex(ValueError, "resume_cursors"):
            self._distributor()
        with open(path) as f:
            self.assertEqual(f.read(), "preserve-on-rejection")

    def test_resume_cursor_beyond_source_history_fails_loudly(self):
        self._publish(2)
        with self.assertRaisesRegex(ValueError, "exceeds the complete source history"):
            self._distributor(resume_cursors={"trainer-a": 3})

    def test_durable_resume_cursor_requires_exact_sample_identity_prefix(self):
        refs = [_ref(i) for i in range(4)]
        self.assertEqual(
            durable_prefix_cursor(refs, [refs[0].sample_id, refs[1].sample_id]), 2
        )
        with self.assertRaisesRegex(ValueError, "exact source prefix"):
            durable_prefix_cursor(refs, [refs[0].sample_id, refs[2].sample_id])
        with self.assertRaisesRegex(ValueError, "absent from source"):
            durable_prefix_cursor(refs, ["unknown"])
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            durable_prefix_cursor([refs[0], refs[0]], [refs[0].sample_id])

    def test_explicit_consumer_failure_poisons_all_and_cleans_pending_once(self):
        refs = self._publish(3, close=False)
        abandoned = []
        dist = self._distributor(
            consumers=("trainer-a", "trainer-b"),
            on_abandoned=lambda batch, error: abandoned.extend(batch),
        )
        dist.pump()
        failed = BroadcastSubscriptionChannel(dist.subscription_paths["trainer-b"])
        failed.report_failure("injected trainer failure")
        with self.assertRaisesRegex(RuntimeError, "trainer-b"):
            dist.pump()
        self.assertEqual(
            [ref.sample_id for ref in abandoned], [ref.sample_id for ref in refs]
        )
        dist.stop()
        self.assertEqual(len(abandoned), 3)
        for path in dist.subscription_paths.values():
            reader = BroadcastSubscriptionChannel(path)
            with self.assertRaisesRegex(RuntimeError, "broadcast-ref-distributor"):
                reader.poll()

    def test_failure_cleanup_includes_source_refs_not_yet_broadcast(self):
        refs = self._publish(8, close=False)
        abandoned = []
        dist = self._distributor(
            consumers=("trainer-a", "trainer-b"),
            max_inflight_refs=2,
            on_abandoned=lambda batch, error: abandoned.extend(batch),
        )
        dist.pump()
        self.assertEqual(dist.stats()["broadcasted"], 2)
        BroadcastSubscriptionChannel(
            dist.subscription_paths["trainer-a"]
        ).report_failure("early exit")
        with self.assertRaisesRegex(RuntimeError, "early exit"):
            dist.pump()
        self.assertEqual(
            [ref.sample_id for ref in abandoned], [ref.sample_id for ref in refs]
        )

    def test_reclaim_callback_failure_never_advances_source(self):
        refs = self._publish(2)
        abandoned = []

        def fail_reclaim(batch):
            self.assertEqual(
                [ref.sample_id for ref in batch], [r.sample_id for r in refs]
            )
            raise RuntimeError("injected owner reclaim failure")

        dist = self._distributor(
            on_global_consumed=fail_reclaim,
            on_abandoned=lambda batch, error: abandoned.extend(batch),
        )
        dist.pump()
        dist.pump()
        queue = self._queue(dist.subscription_paths["trainer-a"])
        queue.ack(queue.get(2))
        with self.assertRaisesRegex(RuntimeError, "owner reclaim failure"):
            dist.pump()
        self.assertEqual(self.producer.consumed_remote(), 0)
        self.assertEqual(
            [ref.sample_id for ref in abandoned], [r.sample_id for r in refs]
        )

    def test_background_run_finishes_and_joins_without_residual_thread(self):
        refs = self._publish(3)
        dist = self._distributor()
        dist.start()
        deadline = time.monotonic() + 2.0
        while not dist.health()["outputs_closed"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        queue = self._queue(dist.subscription_paths["trainer-a"])
        queue.ack(queue.get(3))
        self.assertTrue(dist.wait(timeout_s=2.0))
        self.assertTrue(dist.finished)
        self.assertFalse(dist.health()["thread_alive"])
        self.assertEqual(self.producer.consumed_remote(), len(refs))

    def test_partial_publish_failure_is_never_visible_as_success(self):
        refs = self._publish(1, close=False)
        abandoned = []

        def channel_factory(path):
            if path.endswith("trainer-b.jsonl"):
                return _FailingPublishChannel(path)
            return StreamingRefChannel(path)

        dist = self._distributor(
            consumers=("trainer-a", "trainer-b", "trainer-c"),
            channel_factory=channel_factory,
            on_abandoned=lambda batch, error: abandoned.extend(batch),
        )
        with self.assertRaisesRegex(OSError, "subscription append failure"):
            dist.pump()
        self.assertEqual(dist.stats()["broadcasted"], 0)
        self.assertEqual(dist.stats()["partial_publish_failures"], 1)
        self.assertEqual([ref.sample_id for ref in abandoned], [refs[0].sample_id])
        self.assertEqual(self.producer.consumed_remote(), 0)

    def test_producer_failure_is_terminal_and_never_clean_eof(self):
        dist = BroadcastRefDistributor(
            _FailingSource(self.source_path),
            self.outputs,
            ("trainer-a", "trainer-b"),
            poll_s=0.0,
        )
        with self.assertRaisesRegex(OSError, "producer failure"):
            dist.pump()
        self.assertFalse(dist.finished)
        self.assertEqual(dist.health()["state"], "failed")
        for path in dist.subscription_paths.values():
            self.assertFalse(os.path.exists(path + ".closed"))

    def test_invalid_consumer_identity_and_resume_mapping_rejected(self):
        for ids in ((), ("../unsafe",), ("duplicate", "duplicate")):
            with self.assertRaises(ValueError):
                self._distributor(consumers=ids)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self._distributor(consumers=("a", "b"), resume_cursors={"a": 0})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._distributor(resume_cursors={"trainer-a": -1})


class TestStreamingQueueIdempotency(unittest.TestCase):
    def test_duplicate_ack_and_terminal_failure_advance_once(self):
        root = tempfile.mkdtemp(prefix="stream_terminal_")
        path = os.path.join(root, "refs.jsonl")
        producer = StreamingRefChannel(path)
        refs = [_ref(0), _ref(1)]
        producer.publish_many(refs)
        producer.close()
        queue = StreamingRefQueue(StreamingRefChannel(path), poll_s=0.0)
        got = queue.get(2)
        queue.ack(got[:1])
        queue.ack(got[:1])
        queue.fail(got[1:], reason="terminal", retryable=False)
        queue.fail(got[1:], reason="duplicate", retryable=False)
        self.assertEqual(producer.consumed_remote(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
