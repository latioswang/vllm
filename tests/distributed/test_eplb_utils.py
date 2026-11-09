# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.eplb import async_worker, eplb_state
from vllm.distributed.eplb.async_worker import AsyncEplbExecutor
from vllm.distributed.eplb.eplb_state import (
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
)
from vllm.platforms import current_platform
from vllm.v1.worker.sentinel.eplb_redistribute import reset_eplb_async_state

from .eplb_utils import (
    ASYNC_JOB_TIMEOUT_S,
    async_cycle_running,
    finish_async_cycle,
    make_async_eplb_state,
    start_async_rearrangement,
    wait_for_async_jobs,
)


def _make_model_state(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    phy2log_storage: torch.Tensor | None = None,
) -> MagicMock:
    """Build a minimal EplbModelState mock with its map tensors."""
    state = MagicMock()
    state.physical_to_logical_map = phy2log
    state.physical_to_logical_map_buffer = (
        phy2log if phy2log_storage is None else phy2log_storage
    )
    state.logical_to_physical_map = log2phy
    state.logical_replica_count = logcnt
    return state


def test_commit_eplb_maps_shape_change():
    """When the number of physical experts changes, resize the active map within its
    preallocated storage.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    # Build current state tensors
    storage = torch.full((num_layers, num_physical + 2), -1, dtype=torch.long)
    model_state = _make_model_state(
        phy2log=storage[:, :num_physical],
        phy2log_storage=storage,
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    # The new map has two more physical experts. These new physical experts will
    # automatically map to the first two logical experts
    new_phy2log_larger = (
        (torch.arange(num_physical + 2, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
    )
    _commit_eplb_maps(model_state, new_phy2log_larger)

    # Check that the number of physical experts has been updated and that the values
    # match
    assert model_state.physical_to_logical_map.shape[1] == num_physical + 2
    assert model_state.physical_to_logical_map.data_ptr() == storage.data_ptr()
    assert torch.equal(model_state.physical_to_logical_map, new_phy2log_larger)


def test_commit_eplb_maps_for_layer_logical_padding():
    """Test that logical_to_physical_map is padded with -1 to fill the
    pre-allocated slots when the new map has fewer replicas than the max.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = (
        (torch.arange(num_physical, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
        .contiguous()
    )
    layer = 0
    _commit_eplb_maps_for_layer(model_state, new_phy2log[layer], layer)

    assert torch.all(model_state.logical_to_physical_map[layer, :, 2] == -1)


def test_commit_eplb_maps_for_layer_shape_assert():
    """Test that a mismatched number of physical experts triggers an assertion error."""
    num_layers, num_logical, num_physical = 2, 4, 6

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full((num_layers, num_logical, 2), -1, dtype=torch.long),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )
    bad_phy2log = torch.zeros(num_layers, num_physical + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        _commit_eplb_maps_for_layer(model_state, bad_phy2log, layer=0)


def test_commit_eplb_maps():
    """Test that all values are copied correctly into model_state."""
    num_layers, num_logical, num_physical, max_replicas = 2, 3, 4, 2

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps(model_state, new_phy2log)

    assert torch.equal(model_state.physical_to_logical_map, new_phy2log)
    assert torch.equal(model_state.logical_to_physical_map, new_log2phy)
    assert torch.equal(model_state.logical_replica_count, new_logcnt)


def test_commit_eplb_maps_for_layer():
    """Test that only the target layer is updated"""
    num_layers, num_logical, max_replicas = 2, 3, 2

    original_phy2log = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    model_state = _make_model_state(
        phy2log=original_phy2log.clone(),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps_for_layer(model_state, new_phy2log[0], layer=0)

    # Layer 0 updated
    assert torch.equal(model_state.physical_to_logical_map[0], new_phy2log[0])
    assert torch.equal(model_state.logical_to_physical_map[0], new_log2phy[0])
    assert torch.equal(model_state.logical_replica_count[0], new_logcnt[0])

    # Layer 1 untouched
    assert torch.equal(model_state.physical_to_logical_map[1], original_phy2log[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_commit_eplb_maps_for_layer_does_not_sync_host():
    """Async EPLB commits a layer from the main thread every step, so the commit
    must not synchronize the host with the GPU."""
    num_layers, num_logical, max_replicas = 2, 3, 2
    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, 4, dtype=torch.long, device="cuda"),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas),
            -1,
            dtype=torch.long,
            device="cuda",
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long, device="cuda"),
    )
    new_phy2log = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _commit_eplb_maps_for_layer(model_state, new_phy2log, layer=0)
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)


@pytest.fixture
def async_eplb(dist_init, monkeypatch):
    return make_async_eplb_state(
        monkeypatch, torch.device(current_platform.device_type), num_models=2
    )


def test_async_executor_runs_jobs_on_a_daemon_thread():
    """The worker thread must not block interpreter exit when a job hangs."""
    if not torch.accelerator.is_available():
        pytest.skip("Async EPLB needs an accelerator")
    executor = AsyncEplbExecutor(torch.accelerator.current_device_index())
    assert executor.thread.daemon

    stream = executor.submit(lambda stream: stream).result(ASYNC_JOB_TIMEOUT_S)
    assert isinstance(stream, torch.Stream)

    error = RuntimeError("job failed")

    def failing_job(stream):
        raise error

    assert executor.submit(failing_job).exception(ASYNC_JOB_TIMEOUT_S) is error


def test_async_eplb_applies_every_layer_of_every_model_in_order(async_eplb):
    run = async_eplb
    start_async_rearrangement(run.state)
    finish_async_cycle(run.state)

    expected = [(m, layer) for m in range(2) for layer in range(2)]
    assert run.transfers == expected
    assert [meta[1] for meta in run.applied] == expected
    for model_state in run.model_states:
        expected_map = torch.arange(4).roll(1).repeat(2, 1)
        assert torch.equal(model_state.physical_to_logical_map.cpu(), expected_map)


def test_async_eplb_failure_raises_until_reset(async_eplb):
    """A failed job reaches step() so fault tolerance can react. No new job is
    scheduled until the reset, after which EPLB resumes."""
    run = async_eplb
    run.fail.add((0, 1))
    start_async_rearrangement(run.state)
    with pytest.raises(RuntimeError) as exc_info:
        finish_async_cycle(run.state)
    assert "injected transfer failure" in str(exc_info.value.__cause__)

    with pytest.raises(RuntimeError) as again:
        run.state.step()
    assert again.value is exc_info.value
    assert run.transfers == [(0, 0)]
    # Elastic EP must not start or drain a cycle on a failed runner either.
    with pytest.raises(RuntimeError) as on_start:
        run.state.async_runner.start(run.model_states, [], run.policy, False)
    assert on_start.value is exc_info.value
    with pytest.raises(RuntimeError) as on_drain:
        run.state.drain_async()
    assert on_drain.value is exc_info.value

    reset_eplb_async_state(SimpleNamespace(eplb_state=run.state))
    assert run.state.async_runner.error is None
    assert not async_cycle_running(run.state)
    start_async_rearrangement(run.state)
    finish_async_cycle(run.state)
    assert run.transfers[-4:] == [(m, layer) for m in range(2) for layer in range(2)]


@pytest.mark.parametrize("failing_step", ["submit", "apply"])
def test_async_eplb_main_thread_failure_reaches_every_rank(
    async_eplb, monkeypatch, failing_step
):
    """A failure outside the worker thread raises at once on this rank, and
    the next step reports it through the all-reduce like a failed job, so the
    other ranks do not wait for this one."""
    run = async_eplb
    error = RuntimeError("injected main-thread failure")

    def fail(*args, **kwargs):
        raise error

    with monkeypatch.context() as patch:
        if failing_step == "submit":
            patch.setattr(AsyncEplbExecutor, "submit", fail)
        else:
            patch.setattr(eplb_state, "move_from_buffer", fail)
        with pytest.raises(RuntimeError) as exc_info:
            start_async_rearrangement(run.state)
            finish_async_cycle(run.state)
    assert exc_info.value is error

    with pytest.raises(RuntimeError, match="Async EPLB failed") as collective:
        run.state.step()
    assert collective.value.__cause__ is error

    reset_eplb_async_state(SimpleNamespace(eplb_state=run.state))
    start_async_rearrangement(run.state)
    finish_async_cycle(run.state)
    assert run.transfers[-4:] == [(m, layer) for m in range(2) for layer in range(2)]


def test_reset_leaves_in_flight_job_stale(async_eplb, monkeypatch):
    """Fault recovery does not wait for a running job. Its result is never
    applied, and no new cycle starts until it finishes."""
    run = async_eplb
    gate = threading.Event()
    run.gates[(0, 0)] = gate
    start_async_rearrangement(run.state)
    wait_for_async_jobs(run.state)
    run.state.step()  # Applies the plan and submits the gated transfer.

    reset_eplb_async_state(SimpleNamespace(eplb_state=run.state))
    assert not async_cycle_running(run.state)
    assert run.state.async_runner.busy

    warn = MagicMock()
    monkeypatch.setattr(async_worker, "STALE_JOB_WARNING_INTERVAL_S", 0)
    monkeypatch.setattr(async_worker.logger, "warning", warn)
    plans_before = run.policy.calls
    start_async_rearrangement(run.state)
    # The next check comes after another interval, not on every step.
    assert run.state.expert_rearrangement_step == 0
    run.state.step()
    assert not async_cycle_running(run.state)
    assert run.policy.calls == plans_before
    assert warn.call_count == 2

    del run.gates[(0, 0)]
    gate.set()
    wait_for_async_jobs(run.state)
    start_async_rearrangement(run.state)
    assert async_cycle_running(run.state)
    finish_async_cycle(run.state)

    stale_transfer = 1
    assert run.applied[0][2] != stale_transfer
    assert [meta[1] for meta in run.applied] == [
        (m, layer) for m in range(2) for layer in range(2)
    ]


def test_drain_discards_in_flight_layer(async_eplb):
    run = async_eplb
    start_async_rearrangement(run.state)
    wait_for_async_jobs(run.state)
    run.state.step()  # Submits the first layer transfer.
    wait_for_async_jobs(run.state)

    run.state.drain_async()
    assert not async_cycle_running(run.state)
    assert run.transfers == [(0, 0)]
    assert run.applied == []


def test_drain_waits_for_stale_job(async_eplb, monkeypatch):
    """Elastic EP rearranges right after a drain, so the drain must also wait
    for a discarded transfer that may still write the expert buffer."""
    run = async_eplb
    gate = threading.Event()
    run.gates[(0, 0)] = gate
    start_async_rearrangement(run.state)
    wait_for_async_jobs(run.state)
    run.state.step()  # Submits the gated transfer.
    reset_eplb_async_state(SimpleNamespace(eplb_state=run.state))
    runner = run.state.async_runner
    with pytest.raises(AssertionError, match="still running"):
        runner.start(run.model_states, [], run.policy, False)

    warn = MagicMock()
    monkeypatch.setattr(async_worker, "STALE_JOB_WARNING_INTERVAL_S", 0)
    monkeypatch.setattr(async_worker.logger, "warning", warn)
    drain = threading.Thread(target=run.state.drain_async)
    drain.start()
    drain.join(0.5)
    assert drain.is_alive()
    assert "drain still waits" in warn.call_args.args[0]
    gate.set()
    drain.join(ASYNC_JOB_TIMEOUT_S)
    assert not drain.is_alive()
    assert not runner.busy
    assert run.applied == []
