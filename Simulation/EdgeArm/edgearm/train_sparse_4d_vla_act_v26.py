"""Auditable episode-split training for the V26 sparse 4D-VLA ACT core.

This trainer currently accepts only deterministic high-resolution scratch-RL
replay artifacts.  Human-physical and human-simulation adapters must satisfy
the same policy schema before they can enter a later multisource mixture.  The
trainer reports offline action errors only; it never converts them into a
closed-loop or real-robot success-rate claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset

from .causal_4d_act_v1 import masked_action_chunk_loss
from .ppo_utils_v1 import finite_module_parameters_v1, state_dict_sha256_v1
from .sparse_4d_vla_act_v26 import (
    SPARSE_4D_VLA_ACT_INPUT_KEYS_V26,
    Sparse4DVLAConfigV26,
    Sparse4DVLAACTV26,
    sparse_4d_vla_config_preset_v26,
)
from .sparse_4d_vla_dataset_v26 import Sparse4DVLAReplayDatasetV26
from .trisource_contract_v26 import SIM_RL_SCRATCH_SOURCE_V26


SPARSE_4D_VLA_TRAINER_FORMAT_V26 = "edgearm-v26-sparse-4d-vla-act-training-v1"
SPARSE_4D_VLA_CHECKPOINT_FORMAT_V26 = "edgearm-v26-sparse-4d-vla-act-checkpoint-v1"
OFFLINE_SUCCESS_BOUNDARY_V26 = (
    "offline action error does not measure closed-loop simulation or physical success"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    options["indent" if pretty else "separators"] = 2 if pretty else (",", ":")
    return (json.dumps(payload, **options) + "\n").encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _canonical_bytes(payload, pretty=True))


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _source_hashes_v26() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    paths = (
        Path(__file__).resolve(),
        directory / "sparse_4d_vla_act_v26.py",
        directory / "sparse_4d_vla_dataset_v26.py",
        directory / "trisource_contract_v26.py",
        directory / "replay_rl_multimodal_v26.py",
        directory / "causal_4d_act_v1.py",
        directory / "vla_data.py",
    )
    return {path.name: _sha256_file(path) for path in paths}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("V26 requested MPS but it is unavailable")
        return "mps"
    if device != "cpu":
        raise ValueError("V26 device must be one of auto/cpu/mps")
    return device


@dataclass(frozen=True, slots=True)
class Sparse4DVLATrainingConfigV26:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    validation_episode_fraction: float = 0.20
    current_action_loss_weight: float = 0.50
    seed: int = 26_100_000
    num_workers: int = 0

    def validate(self) -> None:
        integers = (
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "seed",
            "num_workers",
        )
        for name in integers:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"V26 {name} must be an integer")
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("V26 epochs/batch/accumulation must be positive")
        if self.seed < 0 or self.num_workers < 0:
            raise ValueError("V26 seed/workers must be non-negative")
        numerics = (
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip_norm,
            self.validation_episode_fraction,
            self.current_action_loss_weight,
        )
        if any(not math.isfinite(float(value)) for value in numerics):
            raise ValueError("V26 training numerics must be finite")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("V26 learning rate/gradient clip must be positive")
        if self.weight_decay < 0.0 or self.current_action_loss_weight < 0.0:
            raise ValueError("V26 weight decay/current-action weight cannot be negative")
        if not 0.0 < self.validation_episode_fraction < 0.5:
            raise ValueError("V26 validation episode fraction must be in (0,0.5)")


class DeterministicEpochSamplerV26(Sampler[int]):
    def __init__(self, dataset: Dataset[Any], *, seed: int, epoch: int) -> None:
        if len(dataset) < 1 or type(seed) is not int or type(epoch) is not int:
            raise ValueError("V26 deterministic sampler arguments are invalid")
        self.length = len(dataset)
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.length, generator=generator).tolist()


def deterministic_episode_partition_v26(
    episode_keys: Sequence[str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[frozenset[str], frozenset[str]]:
    """Hash-partition unique episode identities; never split individual rows."""

    keys = tuple(episode_keys)
    if len(keys) < 2 or len(set(keys)) != len(keys) or any(not key for key in keys):
        raise ValueError("V26 partition requires at least two unique episode keys")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 0.5:
        raise ValueError("V26 validation fraction is invalid")
    if type(seed) is not int or seed < 0:
        raise ValueError("V26 partition seed must be non-negative")
    ranked = sorted(
        keys,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    validation_count = min(
        len(ranked) - 1,
        max(1, int(round(len(ranked) * validation_fraction))),
    )
    validation = frozenset(ranked[:validation_count])
    train = frozenset(ranked[validation_count:])
    if not train or not validation or train & validation or train | validation != set(keys):
        raise RuntimeError("V26 episode partition violated disjoint union")
    return train, validation


def adapt_sparse_4d_vla_batch_v26(
    batch: Mapping[str, Any],
    *,
    device: str | torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if not isinstance(batch, Mapping) or not isinstance(batch.get("policy_inputs"), Mapping):
        raise TypeError("V26 DataLoader batch lacks nested policy_inputs")
    raw_inputs = batch["policy_inputs"]
    if frozenset(raw_inputs) != SPARSE_4D_VLA_ACT_INPUT_KEYS_V26:
        raise ValueError("V26 batch policy input boundary changed")
    target = batch.get("action_chunk")
    mask = batch.get("action_chunk_mask")
    if not isinstance(target, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("V26 action labels must be tensors")

    def prepare(value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise TypeError("V26 batch fields must enter the adapter as CPU tensors")
        if value.is_floating_point():
            value = value.to(dtype=torch.float32)
        return value.to(device=device)

    inputs = {name: prepare(raw_inputs[name]) for name in SPARSE_4D_VLA_ACT_INPUT_KEYS_V26}
    prepared_target = prepare(target)
    prepared_mask = prepare(mask)
    if prepared_target.dtype != torch.float32 or prepared_mask.dtype != torch.bool:
        raise ValueError("V26 target/mask dtypes changed")
    if not bool(prepared_mask[:, 0].all().item()):
        raise ValueError("V26 every batch item requires a current action label")
    return inputs, prepared_target, prepared_mask


def _build_datasets_and_split(
    replay_paths: Sequence[Path],
    *,
    model_config: Sparse4DVLAConfigV26,
    training_config: Sparse4DVLATrainingConfigV26,
) -> tuple[
    list[Sparse4DVLAReplayDatasetV26],
    Subset[Any],
    Subset[Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    resolved = tuple(Path(path).expanduser().resolve() for path in replay_paths)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("V26 training requires unique replay paths")
    datasets: list[Sparse4DVLAReplayDatasetV26] = []
    source_inventory: list[dict[str, Any]] = []
    try:
        for path in resolved:
            sha256 = _sha256_file(path)
            dataset = Sparse4DVLAReplayDatasetV26(path, model_config=model_config)
            datasets.append(dataset)
            source_inventory.append(
                {
                    "path": str(path),
                    "sha256": sha256,
                    "source_type": SIM_RL_SCRATCH_SOURCE_V26,
                    "action_supervision_samples": len(dataset),
                    "episode_count": len(np.unique(dataset.source_episode_ids)),
                    "expert_calls": 0,
                    "behavior_cloning_steps": 0,
                    "physical_samples": 0,
                }
            )
        concatenated = ConcatDataset(datasets)
        sample_episode_keys: list[str] = []
        episode_keys: set[str] = set()
        for dataset, inventory in zip(datasets, source_inventory, strict=True):
            file_identity = inventory["sha256"]
            for anchor_row in dataset.anchor_rows:
                source_episode = int(dataset.source_episode_ids[int(anchor_row)])
                key = f"{file_identity}:{source_episode}"
                sample_episode_keys.append(key)
                episode_keys.add(key)
        if len(sample_episode_keys) != len(concatenated):
            raise RuntimeError("V26 concatenated sample/episode identities diverged")
        train_keys, validation_keys = deterministic_episode_partition_v26(
            sorted(episode_keys),
            validation_fraction=training_config.validation_episode_fraction,
            seed=training_config.seed,
        )
        train_indices = [index for index, key in enumerate(sample_episode_keys) if key in train_keys]
        validation_indices = [
            index for index, key in enumerate(sample_episode_keys) if key in validation_keys
        ]
        if not train_indices or not validation_indices:
            raise RuntimeError("V26 episode split produced an empty sample subset")
        split_audit = {
            "format": "edgearm-v26-episode-hash-split-v1",
            "row_random_split": False,
            "episode_identity_leakage": False,
            "train_episode_count": len(train_keys),
            "validation_episode_count": len(validation_keys),
            "train_sample_count": len(train_indices),
            "validation_sample_count": len(validation_indices),
            "train_episode_keys_sha256": _canonical_sha256(sorted(train_keys)),
            "validation_episode_keys_sha256": _canonical_sha256(sorted(validation_keys)),
        }
        return (
            datasets,
            Subset(concatenated, train_indices),
            Subset(concatenated, validation_indices),
            source_inventory,
            split_audit,
        )
    except Exception:
        for dataset in datasets:
            dataset.close()
        raise


def _loader(
    dataset: Dataset[Any],
    *,
    config: Sparse4DVLATrainingConfigV26,
    epoch: int,
    training: bool,
) -> DataLoader[Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + epoch + (0x2600 if training else 0x2700))
    sampler = DeterministicEpochSamplerV26(dataset, seed=config.seed, epoch=epoch) if training else None
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        generator=generator,
        persistent_workers=bool(config.num_workers),
    )


def _run_training_epoch(
    model: Sparse4DVLAACTV26,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset[Any],
    *,
    config: Sparse4DVLATrainingConfigV26,
    epoch: int,
    device: str,
) -> dict[str, float | int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    chunk_sum = 0.0
    current_sum = 0.0
    samples = 0
    optimizer_steps = 0
    maximum_gradient_norm = 0.0
    loader = _loader(dataset, config=config, epoch=epoch, training=True)
    for batch_index, batch in enumerate(loader):
        inputs, target, mask = adapt_sparse_4d_vla_batch_v26(batch, device=device)
        prediction = model(inputs)
        chunk_loss = masked_action_chunk_loss(prediction, target, mask)
        current_loss = torch.abs(prediction[:, 0] - target[:, 0]).mean()
        loss = chunk_loss + config.current_action_loss_weight * current_loss
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("V26 sparse 4D-VLA training loss became non-finite")
        (loss / config.gradient_accumulation_steps).backward()
        batch_samples = int(target.shape[0])
        samples += batch_samples
        loss_sum += float(loss.item()) * batch_samples
        chunk_sum += float(chunk_loss.item()) * batch_samples
        current_sum += float(current_loss.item()) * batch_samples
        final = batch_index + 1 == len(loader)
        if (batch_index + 1) % config.gradient_accumulation_steps == 0 or final:
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.item()))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    if samples < 1 or optimizer_steps < 1 or not finite_module_parameters_v1(model):
        raise RuntimeError("V26 sparse 4D-VLA epoch produced no valid optimizer work")
    return {
        "train_samples": samples,
        "train_total_loss": loss_sum / samples,
        "train_masked_chunk_l1": chunk_sum / samples,
        "train_current_action_l1": current_sum / samples,
        "optimizer_steps": optimizer_steps,
        "maximum_preclip_gradient_norm": maximum_gradient_norm,
    }


def _run_validation_epoch(
    model: Sparse4DVLAACTV26,
    dataset: Dataset[Any],
    *,
    config: Sparse4DVLATrainingConfigV26,
    epoch: int,
    device: str,
) -> dict[str, float | int]:
    model.eval()
    chunk_numerator = 0.0
    chunk_denominator = 0
    current_numerator = 0.0
    current_denominator = 0
    samples = 0
    with torch.inference_mode():
        for batch in _loader(dataset, config=config, epoch=epoch, training=False):
            inputs, target, mask = adapt_sparse_4d_vla_batch_v26(batch, device=device)
            prediction = model(inputs)
            absolute = torch.abs(prediction - target)
            selector = mask[:, :, None].expand_as(absolute)
            chunk_numerator += float(absolute.masked_select(selector).sum().item())
            chunk_denominator += int(selector.sum().item())
            current_numerator += float(absolute[:, 0].sum().item())
            current_denominator += int(absolute[:, 0].numel())
            samples += int(target.shape[0])
    if samples < 1 or chunk_denominator < 1 or current_denominator < 1:
        raise RuntimeError("V26 validation produced no supervised samples")
    return {
        "validation_samples": samples,
        "validation_masked_chunk_l1": chunk_numerator / chunk_denominator,
        "validation_current_action_l1": current_numerator / current_denominator,
    }


def train_sparse_4d_vla_act_v26(
    *,
    replay_paths: Sequence[Path],
    output_directory: Path,
    model_config: Sparse4DVLAConfigV26 | None = None,
    training_config: Sparse4DVLATrainingConfigV26 | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    model_cfg = model_config or Sparse4DVLAConfigV26()
    train_cfg = training_config or Sparse4DVLATrainingConfigV26()
    train_cfg.validate()
    resolved_device = _resolve_device(device)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"V26 training output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run_state.json"
    failure_path = output / "failure.json"
    _atomic_json(
        state_path,
        {
            "format": SPARSE_4D_VLA_TRAINER_FORMAT_V26,
            "status": "running",
            "phase": "preflight",
            "updated_at_utc": _utc_now(),
            "production_admission": False,
        },
    )

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed % (2**32))
    datasets: list[Sparse4DVLAReplayDatasetV26] = []
    try:
        datasets, train_dataset, validation_dataset, inventory, split_audit = _build_datasets_and_split(
            replay_paths,
            model_config=model_cfg,
            training_config=train_cfg,
        )
        source_hashes = _source_hashes_v26()
        run_plan = {
            "format": SPARSE_4D_VLA_TRAINER_FORMAT_V26,
            "created_at_utc": _utc_now(),
            "model_config": asdict(model_cfg),
            "training_config": asdict(train_cfg),
            "resolved_device": resolved_device,
            "source_inventory": inventory,
            "episode_split_audit": split_audit,
            "source_code_hashes": source_hashes,
            "policy_input_keys": sorted(SPARSE_4D_VLA_ACT_INPUT_KEYS_V26),
            "training_label_keys": ["action_chunk", "action_chunk_mask"],
            "source_types_currently_executed": ["sim_rl_scratch"],
            "source_types_not_yet_integrated": ["human_physical", "human_simulation"],
            "offline_success_rate_claimed": False,
            "success_boundary": OFFLINE_SUCCESS_BOUNDARY_V26,
            "physical_samples": 0,
            "physical_trials": 0,
            "production_admission": False,
        }
        run_plan["run_plan_sha256"] = _canonical_sha256(run_plan)
        _atomic_json(output / "run_plan.json", run_plan)

        model = Sparse4DVLAACTV26(model_cfg).to(resolved_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        metrics: list[dict[str, Any]] = []
        total_optimizer_steps = 0
        for epoch in range(train_cfg.epochs):
            pre_hash = state_dict_sha256_v1(model.state_dict())
            train_metrics = _run_training_epoch(
                model,
                optimizer,
                train_dataset,
                config=train_cfg,
                epoch=epoch,
                device=resolved_device,
            )
            validation_metrics = _run_validation_epoch(
                model,
                validation_dataset,
                config=train_cfg,
                epoch=epoch,
                device=resolved_device,
            )
            total_optimizer_steps += int(train_metrics["optimizer_steps"])
            record = {
                "format": "edgearm-v26-sparse-4d-vla-epoch-metrics-v1",
                "epoch_index": epoch,
                "epoch_seed": train_cfg.seed + epoch,
                **train_metrics,
                **validation_metrics,
                "pre_model_state_sha256": pre_hash,
                "post_model_state_sha256": state_dict_sha256_v1(model.state_dict()),
                "closed_loop_success_rate": None,
                "closed_loop_success_rate_claimed": False,
                "physical_success_rate": None,
                "physical_success_rate_claimed": False,
            }
            metrics.append(record)
            checkpoint = {
                "format": SPARSE_4D_VLA_CHECKPOINT_FORMAT_V26,
                "run_plan": run_plan,
                "run_plan_sha256": run_plan["run_plan_sha256"],
                "model_metadata": model.metadata(),
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "model_state_sha256": record["post_model_state_sha256"],
                "optimizer_state": optimizer.state_dict(),
                "epoch_completed": epoch + 1,
                "optimizer_steps": total_optimizer_steps,
                "metrics": metrics,
                "production_admission": False,
            }
            _atomic_torch(output / "checkpoints" / "latest.pt", checkpoint)
            _atomic_bytes(
                output / "metrics.jsonl",
                b"".join(_canonical_bytes(item) for item in metrics),
            )
            _atomic_json(
                state_path,
                {
                    "format": SPARSE_4D_VLA_TRAINER_FORMAT_V26,
                    "status": "running",
                    "phase": "training",
                    "updated_at_utc": _utc_now(),
                    "epoch_completed": epoch + 1,
                    "epochs_total": train_cfg.epochs,
                    "optimizer_steps": total_optimizer_steps,
                    "production_admission": False,
                },
            )
        summary = {
            "format": "edgearm-v26-sparse-4d-vla-training-summary-v1",
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "epochs_completed": train_cfg.epochs,
            "optimizer_steps": total_optimizer_steps,
            "final_model_state_sha256": state_dict_sha256_v1(model.state_dict()),
            "latest_metrics": metrics[-1],
            "model_metadata": model.metadata(),
            "source_inventory": inventory,
            "episode_split_audit": split_audit,
            "closed_loop_success_rate": None,
            "closed_loop_success_rate_claimed": False,
            "physical_success_rate": None,
            "physical_success_rate_claimed": False,
            "success_boundary": OFFLINE_SUCCESS_BOUNDARY_V26,
            "remaining_acceptance_gates": [
                "fresh closed-loop held-out simulation evaluation",
                "AQ16 physical intrinsics/extrinsics and latency calibration",
                "UNO-Q depth runtime integration or measured alternative",
                "real stock-gripper trials with exact three-second success",
            ],
            "production_admission": False,
        }
        _atomic_json(output / "summary.json", summary)
        _atomic_json(
            state_path,
            {
                "format": SPARSE_4D_VLA_TRAINER_FORMAT_V26,
                "status": "complete",
                "phase": "complete",
                "updated_at_utc": _utc_now(),
                "epoch_completed": train_cfg.epochs,
                "optimizer_steps": total_optimizer_steps,
                "production_admission": False,
            },
        )
        return summary
    except Exception as error:
        failure = {
            "format": SPARSE_4D_VLA_TRAINER_FORMAT_V26,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "production_admission": False,
        }
        _atomic_json(failure_path, failure)
        _atomic_json(state_path, {**failure, "phase": "failed", "updated_at_utc": _utc_now()})
        raise
    finally:
        for dataset in datasets:
            dataset.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-config-json", type=Path)
    parser.add_argument(
        "--model-scale",
        choices=("pilot", "base_40m"),
        default="pilot",
    )
    parser.add_argument("--training-config-json", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser


def _load_config(path: Path | None, cls: type[Any]) -> Any:
    if path is None:
        return cls()
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V26 configuration JSON must be an object")
    return cls(**payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.model_config_json is not None and args.model_scale != "pilot":
        raise ValueError("V26 model-config-json cannot be combined with a non-pilot preset")
    model_config = (
        _load_config(args.model_config_json, Sparse4DVLAConfigV26)
        if args.model_config_json is not None
        else sparse_4d_vla_config_preset_v26(args.model_scale)
    )
    summary = train_sparse_4d_vla_act_v26(
        replay_paths=args.replay,
        output_directory=args.output_directory,
        model_config=model_config,
        training_config=_load_config(
            args.training_config_json,
            Sparse4DVLATrainingConfigV26,
        ),
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "OFFLINE_SUCCESS_BOUNDARY_V26",
    "SPARSE_4D_VLA_CHECKPOINT_FORMAT_V26",
    "SPARSE_4D_VLA_TRAINER_FORMAT_V26",
    "Sparse4DVLATrainingConfigV26",
    "adapt_sparse_4d_vla_batch_v26",
    "deterministic_episode_partition_v26",
    "train_sparse_4d_vla_act_v26",
]
