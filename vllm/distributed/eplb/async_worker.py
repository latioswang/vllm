# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async EPLB: plan and transfer experts in the background, one layer per step.

`AsyncEplbRunner` lives on the main thread. It submits one job at a time to a
daemon worker thread and acts on the job's result only after every EP rank
has finished the same job. Each job receives everything it touches as
arguments and returns a value or raises, so the worker thread shares no
mutable state with the main thread.
"""

import queue
import threading
import time
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import get_ep_group, get_eplb_group
from vllm.logger import init_logger

from .eplb_communicator import EplbCommunicator
from .eplb_utils import device_stream
from .migration_scheduler import MigrationFlow, schedule_migration_batches_for_layers
from .policy import AbstractEplbPolicy
from .rebalance_execute import TransferMetadata, transfer_layer

if TYPE_CHECKING:
    from .eplb_state import EplbModelState, EplbStats

logger = init_logger(__name__)

AsyncEplbJob = Callable[[torch.Stream], Any]
ApplyLayerFn = Callable[["EplbModelState", int, torch.Tensor, TransferMetadata], None]

STALE_JOB_WARNING_INTERVAL_S = 60.0

# Call sites of the EP all-reduces below. Each all-reduce counts the ranks at
# every call site, so ranks reaching different ones fail loudly on every rank
# instead of silently pairing with each other.
_NUM_TAGS = 3
_POLL_TAG, _READY_TAG, _DRAIN_TAG = range(_NUM_TAGS)


class AsyncEplbExecutor:
    """Runs async EPLB jobs one at a time on a daemon thread.

    The thread is bound to one device and owns one side stream, which every
    job receives as its first argument. A single thread keeps the EPLB group
    collectives of consecutive jobs in submission order on every rank.

    The thread is a daemon so that a job stuck in a transfer with a dead peer
    cannot block interpreter exit. `concurrent.futures.ThreadPoolExecutor` is
    not used because its threads are joined at exit.
    """

    def __init__(self, device_index: int) -> None:
        self._jobs: queue.SimpleQueue[tuple[Future, AsyncEplbJob]] = queue.SimpleQueue()
        self.thread = threading.Thread(
            target=self._run,
            args=(device_index,),
            name="eplb-async-worker",
            daemon=True,
        )
        self.thread.start()

    def submit(self, job: AsyncEplbJob) -> Future:
        future: Future = Future()
        self._jobs.put((future, job))
        return future

    def _run(self, device_index: int) -> None:
        stream: torch.Stream | None = None
        init_error: BaseException | None = None
        try:
            torch.accelerator.set_device_index(device_index)
            stream = torch.Stream(device=torch.accelerator.current_accelerator())
        except BaseException as exc:
            logger.exception("Failed to start the async EPLB worker thread")
            init_error = exc
        while True:
            future, job = self._jobs.get()
            if not future.set_running_or_notify_cancel():
                continue
            if init_error is not None:
                future.set_exception(init_error)
                continue
            try:
                result = job(stream)
            except BaseException as exc:
                logger.exception("Async EPLB job failed")
                future.set_exception(exc)
            else:
                future.set_result(result)


@dataclass
class AsyncEplbPlan:
    """New expert placement for one model, computed by `plan_rebalance`."""

    old_physical_to_logical_map: torch.Tensor
    """Placement when the cycle started, on CPU.
    Shape: (num_moe_layers, num_physical_experts)"""
    new_physical_to_logical_map: torch.Tensor
    """Target placement, on CPU.
    Shape: (num_moe_layers, num_physical_experts)"""
    migration_batches_by_layer: list[list[list[MigrationFlow]]] | None
    """Per-layer transfer schedule when migration batching is enabled."""


@dataclass
class _StaleJob:
    """A job discarded while still running. Its result is never applied."""

    future: Future
    since: float = field(default_factory=time.monotonic)
    last_warning: float = field(default_factory=time.monotonic)

    def should_warn(self) -> bool:
        now = time.monotonic()
        if now - self.last_warning < STALE_JOB_WARNING_INTERVAL_S:
            return False
        self.last_warning = now
        return True


class AsyncEplbRunner:
    """Main-thread driver of async EPLB cycles.

    A cycle is a generator that yields one job at a time: first the plan, then
    one transfer per MoE layer of every model. `poll()` resumes it with the
    job's result once every EP rank has finished that job, so all ranks apply
    the same layer in the same step. Only the main thread touches this object.
    """

    def __init__(self, device_index: int, apply_layer: ApplyLayerFn) -> None:
        """Start the worker thread.

        Args:
            device_index: Device the worker thread binds to.
            apply_layer: Copies a transferred layer out of the expert buffer
                and commits its new maps, on the main stream.

        """
        self._executor = AsyncEplbExecutor(device_index)
        self._apply_layer = apply_layer
        self._cycle: Generator[Future, Any, None] | None = None
        self._job: Future | None = None
        self._stale: _StaleJob | None = None
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        """Set on every rank when a job failed on any rank, until `reset()`."""
        return self._error

    @property
    def cycle_running(self) -> bool:
        return self._cycle is not None

    @property
    def busy(self) -> bool:
        """Whether this rank has a job still running, stale or not."""
        stale = self._stale.future if self._stale is not None else None
        return any(job is not None and not job.done() for job in (self._job, stale))

    def start(
        self,
        model_states: Sequence["EplbModelState"],
        stats: Sequence["EplbStats"],
        policy: type[AbstractEplbPolicy],
        migration_batching: bool,
    ) -> None:
        """Start a cycle. Every EP rank must call this in the same step.

        Raises:
            RuntimeError: A job failed before and `reset()` was not called.

        """
        if self._error is not None:
            raise self._error
        assert self._cycle is None, "An async EPLB cycle is already running."
        assert not self._stale_job_running(), (
            "A discarded async EPLB job is still running."
        )
        self._cycle = self._run_cycle(model_states, stats, policy, migration_batching)
        self._advance(None)

    def poll(self) -> None:
        """Advance the cycle if every rank finished its job. Called by every EP
        rank once per step.

        Raises:
            RuntimeError: A job failed on some rank, now or before `reset()`.

        """
        if self._error is not None:
            raise self._error
        self._warn_about_stale_job()
        if self._cycle is None or not self._all_ranks_done():
            return
        assert self._job is not None
        self._advance(self._job.result())

    def ready_for_new_cycle(self) -> bool:
        """Whether no rank has a cycle or a stale job. Collective.

        A stale transfer still running on one rank must not have its p2p ops
        matched against a new cycle's ops on another rank.
        """
        if self._cycle is not None:
            return False
        _, num_busy = _all_reduce_flags(False, self._stale_job_running(), _READY_TAG)
        return num_busy == 0

    def reset(self) -> None:
        """Clear the error and drop the cycle, for fault recovery.

        Does not wait and is not collective, so it is safe while the EP groups
        are broken. A job still in flight becomes stale: its result is never
        applied, and no rank starts a new cycle until it finishes.
        """
        self._error = None
        self._drop_cycle()

    def drain(self) -> None:
        """Wait until no rank has a job in flight, stale or not, then drop the
        cycle. Collective.

        The in-flight result is discarded. Layers applied earlier in the cycle
        keep their new placement.

        Raises:
            RuntimeError: A job failed on some rank, now or before `reset()`.

        """
        if self._error is not None:
            raise self._error
        draining = self._cycle is not None or self._stale is not None
        if draining:
            logger.info("Draining async EPLB")
        start = last_warning = time.monotonic()
        while True:
            job = self._job
            failed = job is not None and job.done() and job.exception() is not None
            num_failed, num_busy = _all_reduce_flags(failed, self.busy, _DRAIN_TAG)
            if num_failed:
                self._fail(num_failed)
            if num_busy == 0:
                break
            now = time.monotonic()
            if now - last_warning >= STALE_JOB_WARNING_INTERVAL_S:
                last_warning = now
                logger.warning(
                    "Async EPLB drain still waits for a job on %d EP rank(s) "
                    "after %.0f s.",
                    num_busy,
                    now - start,
                )
            time.sleep(0.001)
        self._stale = None
        self._drop_cycle()
        if draining:
            logger.info("Async EPLB drained")

    def _run_cycle(
        self,
        model_states: Sequence["EplbModelState"],
        stats: Sequence["EplbStats"],
        policy: type[AbstractEplbPolicy],
        migration_batching: bool,
    ) -> Generator[Future, Any, None]:
        start = time.perf_counter()
        plans = yield self._submit(
            plan_rebalance,
            policy=policy,
            physical_to_logical_maps=[
                ms.physical_to_logical_map for ms in model_states
            ],
            stats=stats,
            num_local_experts=[
                ms.model.expert_weights[0][0].shape[0] for ms in model_states
            ],
            migration_batching=migration_batching,
        )
        for model_state, plan in zip(model_states, plans):
            num_layers = model_state.model.num_moe_layers
            for layer_idx in range(num_layers):
                transfer_metadata = yield self._submit(
                    transfer_layer_to_buffer,
                    plan=plan,
                    layer_idx=layer_idx,
                    expert_weights=model_state.model.expert_weights[layer_idx],
                    expert_buffer=model_state.expert_buffer,
                    communicator=model_state.communicator,
                    ep_group=get_eplb_group().device_group,
                )
                new_map = plan.new_physical_to_logical_map[layer_idx]
                self._apply_layer(model_state, layer_idx, new_map, transfer_metadata)
            if get_ep_group().rank_in_group == 0:
                logger.info(
                    "Async EPLB rearranged %d/%d layers of %s in %.2f s.",
                    num_layers,
                    num_layers,
                    model_state.model_name,
                    time.perf_counter() - start,
                )
            start = time.perf_counter()

    def _advance(self, result: Any) -> None:
        """Resume the cycle with a job's result, up to its next job."""
        assert self._cycle is not None
        try:
            self._job = self._cycle.send(result)
        except StopIteration:
            self._cycle = self._job = None
        except BaseException as exc:
            # Submitting or applying failed on this rank only. Keep the cycle
            # with a failed job, so that the next collective raises on every
            # rank instead of leaving the other ranks waiting for this one.
            self._job = Future()
            self._job.set_exception(exc)
            raise

    def _submit(self, job: Callable[..., Any], **kwargs: Any) -> Future:
        """Submit a job that may read anything the main stream wrote so far."""
        ready_event = torch.Event()
        ready_event.record()
        return self._executor.submit(partial(job, ready_event=ready_event, **kwargs))

    def _all_ranks_done(self) -> bool:
        """Whether every rank finished the cycle's job. Collective.

        Raises on every rank if the job failed on any rank.
        """
        assert self._job is not None
        done = self._job.done()
        error = self._job.exception() if done else None
        num_failed, num_busy = _all_reduce_flags(error is not None, not done, _POLL_TAG)
        if num_failed:
            self._fail(num_failed)
        return num_busy == 0

    def _fail(self, num_failed: int) -> None:
        """Store and raise the error after a job failed on some rank."""
        error = self._job.exception() if self._job and self._job.done() else None
        failure = RuntimeError(
            f"Async EPLB failed on {num_failed} rank(s); see the logs of those ranks."
        )
        failure.__cause__ = error
        self._error = failure
        self._drop_cycle()
        raise failure

    def _drop_cycle(self) -> None:
        if self._cycle is not None:
            self._cycle.close()
        if self._job is not None and not self._job.done():
            assert self._stale is None
            self._stale = _StaleJob(self._job)
        self._cycle = self._job = None

    def _stale_job_running(self) -> bool:
        """Whether a stale job still runs. Forgets it once it finished."""
        if self._stale is not None and self._stale.future.done():
            self._stale = None
        return self._stale is not None

    def _warn_about_stale_job(self) -> None:
        """Warn now and then that a stale job lingers."""
        stale = self._stale
        if self._stale_job_running() and stale is not None and stale.should_warn():
            logger.warning(
                "Discarded async EPLB job on EP rank %d is still running after "
                "%.0f s; no rebalance starts until it finishes. Serving "
                "continues on the current expert placement.",
                get_ep_group().rank_in_group,
                time.monotonic() - stale.since,
            )


def _all_reduce_flags(failed: bool, busy: bool, tag: int) -> tuple[int, int]:
    """Count the EP ranks whose async EPLB job failed or is still running.

    Raises:
        RuntimeError: Ranks called this from different call sites.

    """
    ep_group = get_ep_group()
    flags = torch.zeros(2 + _NUM_TAGS, dtype=torch.int32)
    flags[0], flags[1], flags[2 + tag] = int(failed), int(busy), 1
    # cpu_all_reduce_ only sums, so count failed and busy ranks.
    ep_group.cpu_all_reduce_(flags)
    num_failed, num_busy, *ranks_per_tag = flags.tolist()
    if ranks_per_tag[tag] != ep_group.world_size:
        raise RuntimeError(
            f"Async EPLB all-reduce mismatch: this rank used tag {tag}, but the "
            f"ranks per tag were {ranks_per_tag}."
        )
    return num_failed, num_busy


def plan_rebalance(
    stream: torch.Stream,
    *,
    ready_event: torch.Event,
    policy: type[AbstractEplbPolicy],
    physical_to_logical_maps: Sequence[torch.Tensor],
    stats: Sequence["EplbStats"],
    num_local_experts: Sequence[int],
    migration_batching: bool,
) -> list[AsyncEplbPlan]:
    """Compute the new placement for every model.

    Args:
        stream: The worker thread's side stream.
        ready_event: Recorded on the main stream after the inputs were written.
        policy: EPLB policy that computes the new placement.
        physical_to_logical_maps: Current placement of each model, on device.
        stats: Load snapshot of each model taken by `EplbState.rearrange`.
        num_local_experts: Physical experts per rank of each model.
        migration_batching: Whether to precompute migration batches.

    Returns:
        One plan per model, in the order of the inputs.

    """
    ready_event.synchronize()
    plans = []
    for physical_to_logical_map, model_stats, local_experts in zip(
        physical_to_logical_maps, stats, num_local_experts
    ):
        with device_stream(stream):
            old_map = physical_to_logical_map.cpu()
            load_window = model_stats.global_expert_load_window.cpu()
        new_map = policy.rebalance_experts(
            load_window,
            model_stats.num_replicas,
            model_stats.num_groups,
            model_stats.num_nodes,
            model_stats.num_gpus,
            old_map,
        )
        assert new_map.device == torch.device("cpu")
        migration_batches = None
        if migration_batching:
            # Covers scheduling for every MoE layer of the model.
            with torch.profiler.record_function("eplb: schedule migration batches"):
                migration_batches = schedule_migration_batches_for_layers(
                    local_experts, old_map.numpy(), new_map.numpy()
                )
        plans.append(AsyncEplbPlan(old_map, new_map, migration_batches))
    return plans


def transfer_layer_to_buffer(
    stream: torch.Stream,
    *,
    ready_event: torch.Event,
    plan: AsyncEplbPlan,
    layer_idx: int,
    expert_weights: Sequence[torch.Tensor],
    expert_buffer: Sequence[torch.Tensor],
    communicator: EplbCommunicator,
    ep_group: ProcessGroup,
) -> TransferMetadata:
    """Fetch the new experts of one layer into `expert_buffer`.

    Args:
        stream: The worker thread's side stream.
        ready_event: Recorded on the main stream after it last read
            `expert_buffer`.
        plan: The model's plan from `plan_rebalance`.
        layer_idx: MoE layer to transfer.
        expert_weights: The layer's expert weights.
        expert_buffer: Receive buffer, free once `ready_event` completes.
        communicator: Communicator for the weight exchange.
        ep_group: The EPLB device group.

    Returns:
        Metadata that `move_from_buffer` needs to apply the layer.

    """
    # Host-issued writes into the buffer (e.g. NIXL RDMA reads) are not
    # stream-ordered, so block until the main stream is done with it.
    ready_event.synchronize()
    communicator.set_stream(stream)
    batches = plan.migration_batches_by_layer
    transfer_metadata = transfer_layer(
        old_layer_indices=plan.old_physical_to_logical_map[layer_idx],
        new_layer_indices=plan.new_physical_to_logical_map[layer_idx],
        expert_weights=expert_weights,
        expert_weights_buffer=expert_buffer,
        communicator=communicator,
        ep_group=ep_group,
        stream=stream,
        layer_idx=layer_idx,
        enable_migration_batching=batches is not None,
        # Only this layer's schedule; transfers never span layers.
        migration_batches=batches[layer_idx] if batches is not None else None,
    )
    # The main thread reads the buffer once the result is visible.
    stream.synchronize()
    return transfer_metadata
