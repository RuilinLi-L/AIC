"""V20 runtime controls, independent of the frozen scientific configuration."""

import os
import time

import torch

from v12_runtime import (
    BatchedFeatureQueue,
    autocast_context,
    loader_options,
    precision_name,
    synchronized_time,
)


RUNTIME_CONFIG_KEYS = frozenset({
    "prefetch_factor", "eval_batch_size", "pin_memory",
    "eta_final_rows", "eta_prior_elapsed_seconds", "eta_prior_elapsed_known",
})


def pin_memory_enabled(device):
    value = os.environ.get("AIC_PIN_MEMORY", "0")
    if value not in ("0", "1"):
        raise ValueError("AIC_PIN_MEMORY must be 0 or 1")
    return device.type == "cuda" and value == "1"


def scientific_config(config):
    """Allow loader tuning; V20 never substitutes an accumulated microbatch."""
    batch = config.get("batch_size", 256)
    accumulation = config.get("gradient_accumulation", 1)
    if (isinstance(batch, bool) or isinstance(accumulation, bool)
            or not isinstance(batch, int) or not isinstance(accumulation, int)
            or (batch, accumulation) != (256, 1)):
        raise ValueError("V20 requires actual batch_size=256 and gradient_accumulation=1")
    normalized = {**config, "batch_size": batch, "gradient_accumulation": accumulation}
    return {key: value for key, value in normalized.items() if key not in RUNTIME_CONFIG_KEYS}


def training_history_seconds(history):
    return sum(float(record.get(key, 0.0)) for record in history
               for key in ("train_seconds", "validation_seconds", "audit_seconds", "supervision_seconds"))


def pipeline_eta(history, *, stage, stop_epoch, train_rows, final_rows,
                 prior_elapsed_seconds=0.0, prior_elapsed_known=True, budget_hours=48.0):
    """Estimate remaining work without making any training or selection decision."""
    if stage not in ("validate", "refit") or len(history) < 2:
        raise ValueError("pipeline ETA requires two measured epochs and a valid stage")
    if train_rows <= 0 or final_rows <= 0:
        raise ValueError("pipeline ETA requires positive row counts")
    recent = history[-2:]
    train_mean = sum(float(record["train_seconds"]) + float(record.get("supervision_seconds", 0)) for record in recent) / len(recent)
    validation_mean = sum(float(record.get("validation_seconds", 0)) for record in recent) / len(recent)
    audited = [record for record in history if int(record["epoch"]) in (1, 2) or int(record["epoch"]) % 2 == 0]
    audit_mean = sum(float(record.get("audit_seconds", 0)) for record in audited[-2:]) / max(len(audited[-2:]), 1)
    completed_epoch = int(history[-1]["epoch"])
    remaining_epochs = max(0, int(stop_epoch) - completed_epoch)
    audit_count = lambda start, stop: sum(epoch in (1, 2) or epoch % 2 == 0 for epoch in range(start, stop + 1))
    stage_remaining = remaining_epochs * (train_mean + (validation_mean if stage == "validate" else 0.0))
    stage_remaining += audit_count(completed_epoch + 1, int(stop_epoch)) * audit_mean
    scale = float(final_rows) / train_rows
    if stage == "validate":
        reserve = 8.0 * 3600
        refit = [epochs * train_mean * scale + audit_count(1, epochs) * audit_mean * scale for epochs in (18, 24)]
        remaining = [stage_remaining + value + reserve for value in refit]
        refit_epochs = [18, 24]
    else:
        reserve = 1.0 * 3600
        refit, refit_epochs = [0.0, 0.0], [int(stop_epoch), int(stop_epoch)]
        remaining = [stage_remaining + reserve] * 2
    measured = float(prior_elapsed_seconds) + training_history_seconds(history)
    pipeline_started = os.environ.get("AIC_PIPELINE_STARTED_AT")
    elapsed = max(measured, time.time() - float(pipeline_started)) if pipeline_started else measured
    total = [elapsed + value for value in remaining]
    budget = float(budget_hours) * 3600
    return {
        "stage": stage, "completed_epoch": completed_epoch, "stop_epoch": int(stop_epoch),
        "measured_history_seconds": measured, "prior_elapsed_known": bool(prior_elapsed_known),
        "remaining_stage_seconds": stage_remaining, "refit_epochs_assumption": refit_epochs,
        "refit_row_scale": scale, "refit_seconds_range": refit,
        "evaluation_prediction_reserve_seconds": reserve,
        "remaining_seconds_range": remaining, "estimated_total_seconds_range": total,
        "budget_hours": float(budget_hours), "budget_exceeded": total[1] > budget,
        "budget_exceeded_at_low_estimate": total[0] > budget,
        "external_dependency_wait_included": bool(pipeline_started),
        "elapsed_wall_clock_seconds": elapsed if pipeline_started else None,
        "budget_clock_source": "pipeline_start" if pipeline_started else "measured_history",
        "future_dependency_wait_seconds": None,
        "assumptions": "recent two measured epochs; 8h joint evaluation reserve includes 24-epoch four-view baseline caches; refit epoch is fixed by selection, not this estimate; "
                       "elapsed queue/preparation/dependency time is included when launched by the pipeline; "
                       "future sibling/dependency waits are unknown, so total is a lower estimate; "
                       "budget warnings never shorten training",
    }


class Progress:
    def __init__(self, stage, device, batches):
        self.stage, self.device, self.batches = stage, device, batches
        self.started = synchronized_time(device)
        self.rows = 0

    def update(self, batch_id, rows):
        self.rows += rows
        if batch_id == 1 or batch_id % 50 == 0 or batch_id == self.batches:
            elapsed = synchronized_time(self.device) - self.started
            memory = ""
            if self.device.type == "cuda":
                memory = (f" allocated_gib={torch.cuda.memory_allocated(self.device) / 2**30:.2f}"
                          f" peak_gib={torch.cuda.max_memory_allocated(self.device) / 2**30:.2f}")
            print(f"v20_progress stage={self.stage} batches={batch_id}/{self.batches} "
                  f"rows={self.rows} images_per_s={self.rows / max(elapsed, 1e-9):.2f}"
                  f" elapsed_s={elapsed:.1f}{memory}", flush=True)
