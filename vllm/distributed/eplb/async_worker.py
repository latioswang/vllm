# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async EPLB: transfer experts in the background, one layer per step.

`AsyncEplbRunner` lives on the main thread. It submits one job at a time to a
daemon worker thread and acts on the result only after every EP rank finished
the same job. A job gets everything it touches as arguments and returns a
value or raises, so the two threads share no mutable state.
"""

import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
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


class AsyncEplbExecutor:
    """Runs async EPLB jobs one at a time on a daemon thread.

    The thread is bound to one device and owns one side stream, which every
    job receives as its first argument. It is a daemon so that a job stuck in
    a transfer with a dead peer cannot block interpreter exit;
    `ThreadPoolExecutor` threads are joined at exit.
    """

    def __init__(self, device_index: int) -> None:
        self._jobs: queue.SimpleQueue[tuple[Future, AsyncEplbJob]] = queue.SimpleQueue()
        self.thread = threading.Thread(
            target=self._run, args=(device_index,), daemon=True
        )
        self.thread.start()

    def submit(self, job: AsyncEplbJob) -> Future:
        future: Future = Future()
        self._jobs.put((future, job))
        return future

    def _run(self, device_index: int) -> None:
        stream: torch.Stream | None = None
        while True:
            future, job = self._jobs.get()
            try:
                if stream is None:
                    torch.accelerator.set_device_index(device_index)
                    device = torch.accelerator.current_accelerator()
                    stream = torch.Stream(device=device)
                future.set_result(job(stream))
            except BaseException as exc:
                logger.exception("Async EPLB job failed")
                future.set_exception(exc)


@dataclass
class AsyncEplbPlan:
    """Old and new placement of one model, on CPU."""

    old_physical_to_logical_map: torch.Tensor
    new_physical_to_logical_map: torch.Tensor
    migration_batches_by_layer: list[list[list[MigrationFlow]]] | None


class AsyncEplbRunner:
    """Main-thread driver of async EPLB cycles.

    A cycle is a list of layers to rearrange. `poll()` applies a layer and
    submits the next one once every EP rank finished the current job, so all
    ranks apply the same layer in the same step.
    """

    def __init__(
        self,
        device_index: int,
        apply_layer: ApplyLayerFn,
        policy: type[AbstractEplbPolicy],
        migration_batching: bool,
    ) -> None:
        self._executor = AsyncEplbExecutor(device_index)
        self._apply_layer = apply_layer
        self._policy = policy
        self._migration_batching = migration_batching
        # Layers of the running cycle that are not applied yet. The job of
        # the first one is running or finished.
        self._todo: deque[tuple[EplbModelState, int]] = deque()
        # The last submitted job. If its cycle was dropped, its result is
        # never applied.
        self._job: Future | None = None
        # Set on every rank by a reset or a failure, which may leave a job
        # running on some rank.
        self._settling = False
        self._error: BaseException | None = None
        self._start_time = 0.0

    @property
    def cycle_running(self) -> bool:
        return bool(self._todo)

    @property
    def busy(self) -> bool:
        """Whether this rank has a job still running, dropped or not."""
        return self._job is not None and not self._job.done()

    def start(self, model_states: list["EplbModelState"]) -> None:
        """Start a cycle. Every EP rank must call this in the same step."""
        if self._error is not None:
            raise self._error
        assert not self.cycle_running and not self.busy
        self._todo.extend(
            (model_state, layer_idx)
            for model_state in model_states
            for layer_idx in range(model_state.model.num_moe_layers)
        )
        self._advance(apply_first=False)

    def poll(self) -> None:
        """Advance the cycle if every rank finished its job. Called by every EP
        rank once per step; raises on every rank if a job failed on any rank.
        """
        if self._error is not None:
            raise self._error
        if self._todo and self._all_ranks_done():
            self._advance(apply_first=True)

    def ready_for_new_cycle(self) -> bool:
        """Whether no rank has a cycle or a dropped job still running.

        A dropped transfer still running on one rank must not have its p2p
        ops matched against a new cycle's ops on another rank.
        """
        if self._todo:
            return False
        if self._settling:
            self._settling = _count_failed_and_busy_ranks(False, self.busy)[1] > 0
        return not self._settling

    def reset(self) -> None:
        """Clear the error and drop the cycle, for fault recovery.

        Not collective, so it is safe while the EP groups are broken. Every
        rank must call it between the same two steps.
        """
        self._error = None
        self._drop_cycle()

    def drain(self) -> None:
        """Run the cycle to its end in lockstep."""
        draining = self.cycle_running
        if draining:
            logger.info("Draining async EPLB worker")
        while True:
            self.poll()
            if not self.cycle_running and self.ready_for_new_cycle():
                break
            time.sleep(0.001)
        if draining:
            logger.info("Async EPLB worker drained")

    def _advance(self, apply_first: bool) -> None:
        """Apply the first layer if its job finished, then submit the next."""
        try:
            plan = None
            if apply_first:
                assert self._job is not None
                model_state, layer_idx = self._todo[0]
                num_layers = model_state.model.num_moe_layers
                plan, transfer_metadata = self._job.result()
                self._apply_layer(
                    model_state,
                    layer_idx,
                    plan.new_physical_to_logical_map[layer_idx],
                    transfer_metadata,
                )
                self._todo.popleft()
                if layer_idx == num_layers - 1 and get_ep_group().rank_in_group == 0:
                    logger.info(
                        "Async EPLB rearranged %d/%d layers of %s in %.2f s.",
                        num_layers,
                        num_layers,
                        model_state.model_name,
                        time.perf_counter() - self._start_time,
                    )
            if not self._todo:
                return
            model_state, layer_idx = self._todo[0]
            if layer_idx == 0:
                # The job of a model's first layer computes the plan.
                plan = None
                self._start_time = time.perf_counter()
            self._job = self._submit(
                transfer_layer_to_buffer,
                plan=plan,
                policy=self._policy,
                eplb_stats=model_state.eplb_stats,
                physical_to_logical_map=model_state.physical_to_logical_map,
                migration_batching=self._migration_batching,
                layer_idx=layer_idx,
                expert_weights=model_state.model.expert_weights[layer_idx],
                expert_buffer=model_state.expert_buffer,
                communicator=model_state.communicator,
                ep_group=get_eplb_group().device_group,
            )
        except BaseException as exc:
            # Keep the cycle with a failed job, so that the next all-reduce
            # raises on every rank instead of leaving the others waiting.
            self._job = Future()
            self._job.set_exception(exc)
            raise

    def _submit(self, job: Callable[..., Any], **kwargs: Any) -> Future:
        # The job waits for this event, so it sees every main-stream write
        # issued so far, e.g. the previous layer's copy out of the buffer.
        ready_event = torch.Event()
        ready_event.record()
        return self._executor.submit(partial(job, ready_event=ready_event, **kwargs))

    def _all_ranks_done(self) -> bool:
        assert self._job is not None
        done = self._job.done()
        error = self._job.exception() if done else None
        num_failed, num_busy = _count_failed_and_busy_ranks(error is not None, not done)
        if num_failed:
            self._error = RuntimeError(
                "Async EPLB worker failed on at least one rank; see the worker logs."
            )
            self._error.__cause__ = error
            self._drop_cycle()
            raise self._error
        return num_busy == 0

    def _drop_cycle(self) -> None:
        if self.busy:
            logger.warning(
                "Dropped an async EPLB job that is still running; no "
                "rebalance starts until it finishes."
            )
        self._todo.clear()
        self._settling = True


def _count_failed_and_busy_ranks(failed: bool, busy: bool) -> tuple[int, int]:
    flags = torch.tensor((int(failed), int(busy)), dtype=torch.int32, device="cpu")
    # cpu_all_reduce_ only sums, so count failed and busy ranks.
    get_ep_group().cpu_all_reduce_(flags)
    num_failed, num_busy = flags.tolist()
    return num_failed, num_busy


def run_rebalance_experts(
    policy: type[AbstractEplbPolicy],
    eplb_stats: "EplbStats",
    physical_to_logical_map_cpu: torch.Tensor,
    stream: torch.Stream,
) -> torch.Tensor:
    # Move the global expert load window to CPU for computation.
    with device_stream(stream):
        global_expert_load_window = eplb_stats.global_expert_load_window.cpu()
    # Compute new expert mappings for the model
    new_physical_to_logical_map = policy.rebalance_experts(
        global_expert_load_window,
        eplb_stats.num_replicas,
        eplb_stats.num_groups,
        eplb_stats.num_nodes,
        eplb_stats.num_gpus,
        physical_to_logical_map_cpu,
    )
    assert new_physical_to_logical_map.device == torch.device("cpu")

    return new_physical_to_logical_map


def transfer_layer_to_buffer(
    stream: torch.Stream,
    *,
    ready_event: torch.Event,
    plan: AsyncEplbPlan | None,
    policy: type[AbstractEplbPolicy],
    eplb_stats: "EplbStats",
    physical_to_logical_map: torch.Tensor,
    migration_batching: bool,
    layer_idx: int,
    expert_weights: Sequence[torch.Tensor],
    expert_buffer: Sequence[torch.Tensor],
    communicator: EplbCommunicator,
    ep_group: ProcessGroup,
) -> tuple[AsyncEplbPlan, TransferMetadata]:
    """Fetch the new experts of one layer into `expert_buffer`.

    The job of a model's first layer gets no `plan` and computes it first.
    """
    # Host-issued writes into expert_buffer (e.g. NIXL RDMA reads) are not
    # stream-ordered, so wait until the main stream is done with it.
    ready_event.synchronize()
    # Set the async worker's CUDA stream on the communicator
    communicator.set_stream(stream)

    if plan is None:
        # Snapshot the physical_to_logical_map and copy it to CPU
        with device_stream(stream):
            physical_to_logical_map_cpu = physical_to_logical_map.cpu()

        new_physical_to_logical_map = run_rebalance_experts(
            policy, eplb_stats, physical_to_logical_map_cpu, stream
        )

        migration_batches_by_layer = None
        if migration_batching:
            profile_name = "eplb: schedule migration batches"
            num_local_experts = expert_weights[0].shape[0]
            # The placement snapshots stay fixed for this rebalance cycle.
            # Precompute all layer schedules once, but execute transfers one
            # layer at a time to preserve the migration lifecycle. This
            # profiling range therefore covers scheduling for every MoE layer.
            with torch.profiler.record_function(profile_name):
                migration_batches_by_layer = schedule_migration_batches_for_layers(
                    num_local_experts,
                    physical_to_logical_map_cpu.numpy(),
                    new_physical_to_logical_map.numpy(),
                )
        plan = AsyncEplbPlan(
            physical_to_logical_map_cpu,
            new_physical_to_logical_map,
            migration_batches_by_layer,
        )

    migration_batches_by_layer = plan.migration_batches_by_layer
    transfer_metadata = transfer_layer(
        old_layer_indices=plan.old_physical_to_logical_map[layer_idx],
        new_layer_indices=plan.new_physical_to_logical_map[layer_idx],
        expert_weights=expert_weights,
        expert_weights_buffer=expert_buffer,
        communicator=communicator,
        ep_group=ep_group,
        stream=stream,
        layer_idx=layer_idx,
        enable_migration_batching=migration_batching,
        migration_batches=(
            # This is only the current layer's schedule; it does not
            # combine transfers from different layers.
            migration_batches_by_layer[layer_idx]
            if migration_batches_by_layer is not None
            else None
        ),
    )

    # Wait until all writes to expert_buffer have finished before making the
    # result visible to the main thread.
    stream.synchronize()
    return plan, transfer_metadata
