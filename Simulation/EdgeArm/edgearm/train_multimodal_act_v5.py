"""Local ACT-inspired action distillation. Offline selection is not task acceptance.

Full passes preserve sample coverage. Hierarchical loss weights balance sources,
position pairs, episodes, temporal-progress bins and command-magnitude bins.
These strata are training metadata, never actor inputs or privileged state.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .candidate_command_contract_v2 import ACTION_CONTRACT
from .multimodal_act_v5 import MultimodalACTV5
from .sparse_4d_vla_act_v26 import Sparse4DVLAConfigV26
from .temporal_input_contract_v3 import INPUT_KEYS, TemporalPackets
from .train_staged_hybrid_contact_sac import _atomic_json
from .train_temporal_spatial_v3 import future_tool_loss


FORMAT = "edgearm-multimodal-act-v5-behavior-distillation"


def batch_to_device(batch, device):
    return {
        k: {name: tensor.to(device) for name, tensor in value.items()}
        if isinstance(value, dict)
        else value.to(device)
        for k, value in batch.items()
    }


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_records(manifest, split):
    rows = json.loads(Path(manifest).read_text())["records"]
    chosen = [r for r in rows if r["split"] == split]
    if not chosen:
        raise ValueError(f"no {split} records in {manifest}")
    for row in chosen:
        if sha256(row["packet"]) != row["packet_sha256"]:
            raise ValueError(f"packet hash mismatch: {row['packet']}")
        if row["source"] not in ("rl", "human"):
            raise ValueError("unreviewed source semantics")
    return chosen


def require_disjoint(train, validation):
    for key in ("packet_sha256", "parent_source_path", "group"):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f"train/validation overlap: {key}")


def balanced_window_weights(dataset):
    """Give every available stratum equal mass inside its parent stratum."""
    buckets = defaultdict(list)
    details = []
    commands = {}
    for ri, record in enumerate(dataset.records):
        with np.load(record["packet"]) as z:
            commands[ri] = z["command"]
    for index, (ri, t) in enumerate(dataset.indices):
        record = dataset.records[ri]
        progress = t / max(record["length"] - 1, 1)
        temporal = "early" if progress < 0.2 else "late" if progress >= 0.8 else "middle"
        magnitude = float(np.abs(commands[ri][t, :5]).mean())
        motion = "small" if magnitude < 0.02 else "medium" if magnitude < 0.1 else "large"
        key = (record["source"], record.get("pair") or "human", ri, temporal, motion)
        buckets[key].append(index)
    children = defaultdict(set)
    for key in buckets:
        for depth in range(1, len(key)):
            children[key[:depth]].add(key[depth])
    sources = {r["source"] for r in dataset.records}
    source_mass = {s: {"rl": 0.8, "human": 0.2}[s] for s in sources}
    total_source_mass = sum(source_mass.values())
    weights = np.zeros(len(dataset.indices), np.float64)
    for key, indices in sorted(buckets.items()):
        mass = source_mass[key[0]] / total_source_mass
        for depth in range(1, len(key)):
            mass /= len(children[key[:depth]])
        weights[indices] = mass / len(indices)
        details.append(
            dict(
                source=key[0],
                pair=key[1],
                episode=key[2],
                progress_bin=key[3],
                magnitude_bin=key[4],
                windows=len(indices),
                probability_mass=mass,
            )
        )
    if not np.isclose(weights.sum(), 1) or not (weights > 0).all():
        raise ValueError("invalid balancing weights")
    # With a uniformly shuffled full pass this is an unbiased weighted objective.
    return weights * len(weights), details


def command_loss(prediction, target, mask, weights, repair=False):
    if not mask[:, 0].all() or not mask.any(1).all():
        raise ValueError("a training window needs a valid current command")
    safe_target = torch.where(mask[..., None], target, 0.0)
    error = (prediction - safe_target).abs().mean(-1)
    chunk = (error * mask).sum(1) / mask.sum(1)
    if repair:
        # Explicit new objective: emphasize executed action and demonstrated hold.
        hold = (target[..., :5].abs().amax(-1) < 0.02) & mask
        hold_error = (error * hold).sum(1) / hold.sum(1).clamp_min(1)
        bounded = weights.clamp(max=4.0)
        return ((chunk + 2 * error[:, 0] + hold_error) * bounded).sum() / bounded.sum().clamp_min(1e-8)
    return ((chunk + error[:, 0]) * weights).mean()


def evaluate_prior(model, dataset, batch_size=4):
    model.eval()
    device = next(model.parameters()).device
    totals = defaultdict(
        lambda: dict(error=0.0, count=0, first=0.0, first_count=0, idle_error=0.0, idle_first=0.0)
    )
    with torch.inference_mode():
        for start in range(0, len(dataset.indices), batch_size):
            ids = list(range(start, min(start + batch_size, len(dataset.indices))))
            b = batch_to_device(dataset.batch(ids), device)
            prediction = model(b["inputs"])
            if not torch.isfinite(prediction).all():
                raise ValueError("nonfinite prior validation")
            prediction = prediction.cpu()
            b["target"], b["mask"] = b["target"].cpu(), b["mask"].cpu()
            for j, index in enumerate(ids):
                source = dataset.records[dataset.indices[index][0]]["source"]
                for key in (source, "pooled"):
                    row = totals[key]
                    valid = b["mask"][j, :, None].expand_as(prediction[j])
                    row["error"] += float((prediction[j] - b["target"][j]).abs()[valid].sum())
                    row["count"] += int(valid.sum())
                    row["idle_error"] += float(b["target"][j].abs()[valid].sum())
                    row["first"] += float((prediction[j, 0] - b["target"][j, 0]).abs().sum())
                    row["idle_first"] += float(b["target"][j, 0].abs().sum())
                    row["first_count"] += 6
    result = {
        k: dict(
            command_mae=v["error"] / v["count"],
            first_command_mae=v["first"] / v["first_count"],
            zero_command_mae=v["idle_error"] / v["count"],
            zero_first_command_mae=v["idle_first"] / v["first_count"],
            labeled_elements=v["count"],
        )
        for k, v in totals.items()
    }
    present = [s for s in ("rl", "human") if s in result]
    mass = {"rl": 0.8, "human": 0.2}
    result["selection_score"] = sum(mass[s] * result[s]["first_command_mae"] for s in present) / sum(
        mass[s] for s in present
    )
    result["posterior_used"] = False
    return result


def warm_start(base_checkpoint, future_checkpoint):
    base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    if (
        base.get("format") != "temporal_spatial_v3_integration"
        or base.get("action_contract") != ACTION_CONTRACT
        or frozenset(base["input_keys"]) != INPUT_KEYS
    ):
        raise ValueError("base checkpoint contract mismatch")
    model = MultimodalACTV5(Sparse4DVLAConfigV26(**base["config"]))
    model.policy.load_state_dict(base["model"])
    future = torch.load(future_checkpoint, map_location="cpu", weights_only=False)
    if future["base_policy_sha256"] != sha256(base_checkpoint):
        raise ValueError("future head belongs to another action policy")
    model.future_head.load_state_dict(future["head"])
    # Preserve the separately audited depth estimate during this action experiment.
    for module in (model.policy.depth_student, model.policy.core.rgbd_encoder, model.policy.next_tool_head):
        module.requires_grad_(False)
    return model


def initial_pose_shortcut_audit(train_rows, val_rows):
    def initial(rows):
        selected = [r for r in rows if r["source"] == "rl"]
        states = []
        for r in selected:
            with np.load(r["packet"]) as z:
                states.append(z["joint"][0, :6])
        return selected, np.asarray(states)

    tr, x = initial(train_rows)
    va, y = initial(val_rows)
    nearest = ((y[:, None] - x[None]) ** 2).sum(-1).argmin(1)
    rows = [
        dict(
            group=r["group"],
            pair=r["pair"],
            predicted_pair=tr[int(k)]["pair"],
            correct=r["pair"] == tr[int(k)]["pair"],
        )
        for r, k in zip(va, nearest)
    ]
    return dict(
        method="1NN raw initial six joint angles; no image or language",
        episodes=rows,
        correct=sum(r["correct"] for r in rows),
        count=len(rows),
        note="diagnostic only, not an actor feature or evidence of language grounding",
    )


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    began = time.time()
    _atomic_json(
        out / "run_state.json", dict(status="preflight", pid=os.getpid(), production_admission=False)
    )
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; CPU fallback forbidden")
        if args.initial_checkpoint:
            ck = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
            if (
                ck.get("format") != FORMAT
                or ck.get("action_contract") != ACTION_CONTRACT
                or frozenset(ck["input_keys"]) != INPUT_KEYS
            ):
                raise ValueError("initial ACT checkpoint contract mismatch")
            model = MultimodalACTV5(Sparse4DVLAConfigV26(**ck["config"]))
            model.load_state_dict(ck["model"])
            for module in (
                model.policy.depth_student,
                model.policy.core.rgbd_encoder,
                model.policy.next_tool_head,
            ):
                module.requires_grad_(False)
        else:
            model = warm_start(args.base_checkpoint, args.future_checkpoint)
        if args.unfreeze_rgb:
            model.policy.core.rgbd_encoder.requires_grad_(True)
        model.to(device)
        tr = read_records(args.train_manifest, "train")
        va = read_records(args.validation_manifest, "validation")
        require_disjoint(tr, va)
        train = TemporalPackets(
            tr, model.config, cache_episodes=args.cache_episodes, include_auxiliary_depth=False
        )
        val = TemporalPackets(
            va, model.config, cache_episodes=args.cache_episodes, include_auxiliary_depth=False
        )
        weights, strata = balanced_window_weights(train)
        _atomic_json(
            out / "data_audit.json",
            dict(
                train_episodes=len(tr),
                validation_episodes=len(va),
                train_sources=dict(Counter(r["source"] for r in tr)),
                train_pairs=dict(Counter(r.get("pair") or "human" for r in tr)),
                strata=strata,
                disjoint=True,
                initial_pose_shortcut=initial_pose_shortcut_audit(tr, va),
                train_manifest_sha256=sha256(args.train_manifest),
                validation_manifest_sha256=sha256(args.validation_manifest),
                holdout_used=False,
                real_human_count=0,
                multiview_available=False,
            ),
        )
        plan = dict(
            format=FORMAT,
            config=asdict(model.config),
            arguments=vars(args),
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            train_windows=len(train.indices),
            validation_windows=len(val.indices),
            action_contract=ACTION_CONTRACT,
            input_keys=sorted(INPUT_KEYS),
            weights=dict(
                prior_chunk=1.0,
                prior_first=1.0,
                posterior_chunk=0.5,
                posterior_first=0.5,
                kl=0.001,
                future_tool_per_0p1m=0.01,
                human_label_quality=0.25,
            ),
            sampling="full shuffled passes; hierarchical source/pair/episode/progress/magnitude loss weighting",
            frozen_modules=(["RGB convolutional encoder"] if not args.unfreeze_rgb else [])
            + ["depth student", "obsolete absolute next-tool head"],
            device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            initial_checkpoint_sha256=sha256(args.initial_checkpoint) if args.initial_checkpoint else None,
            optimizer_reinitialized=True,
            prior="deterministic z=0",
            posterior="training labels only",
            persistent_voxel_memory_admitted=False,
            exact_home_evaluated=False,
            source_lineage="legacy staged RL plus simulated human; not certified scratch or real-human",
            production_admission=False,
            export_admission=False,
            final_vla_acceptance=False,
        )
        _atomic_json(out / "plan.json", plan)
        model.eval()
        with torch.no_grad():
            for data in (train, val):
                for ri in range(len(data.records)):
                    i = next(i for i, (r, _) in enumerate(data.indices) if r == ri)
                    b = batch_to_device(data.batch([i]), device)
                    torch.testing.assert_close(model(b["inputs"]), model.policy(b["inputs"]), atol=0, rtol=0)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}
        baseline = evaluate_prior(model, val, args.batch_size)
        _atomic_json(out / "baseline_validation.json", baseline)
        best = baseline["selection_score"]
        step = 0
        seen = set()

        def save(path, epoch):
            torch.save(
                dict(
                    format=FORMAT,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    config=asdict(model.config),
                    action_contract=ACTION_CONTRACT,
                    input_keys=sorted(INPUT_KEYS),
                    epoch=epoch,
                    step=step,
                    seed=args.seed,
                    torch_rng_state=torch.get_rng_state(),
                    numpy_rng_state=rng.bit_generator.state,
                    cuda_rng_state_all=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    initial_checkpoint_sha256=sha256(args.initial_checkpoint)
                    if args.initial_checkpoint
                    else None,
                    device=str(device),
                    base_policy_sha256=sha256(args.base_checkpoint),
                    future_head_sha256=sha256(args.future_checkpoint),
                    production_admission=False,
                    export_admission=False,
                    exact_home_evaluated=False,
                    final_vla_acceptance=False,
                ),
                path,
            )

        save(out / "checkpoint_best.pt", 0)
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(len(train.indices))
            for start in range(0, len(order), args.batch_size):
                ids = order[start : start + args.batch_size].tolist()
                seen.update(ids)
                b = batch_to_device(train.batch(ids), device)
                w = torch.tensor(weights[ids], dtype=torch.float32, device=device)
                quality = torch.tensor(
                    [1.0 if tr[train.indices[i][0]]["source"] == "rl" else 0.25 for i in ids], device=device
                )
                prediction = model(
                    b["inputs"], return_aux=True, teacher_actions=b["target"], teacher_mask=b["mask"]
                )
                repair = getattr(args, "repair_objective", False)
                prior = command_loss(prediction["action"], b["target"], b["mask"], w * quality, repair)
                posterior = command_loss(prediction["posterior_action"], b["target"], b["mask"], w * quality, repair)
                future = future_tool_loss(
                    prediction["next_tool_xyz"], b["future_tool_xyz"], b["future_tool_mask"]
                )
                loss = prior + 0.5 * posterior + 0.001 * prediction["kl"] + 0.01 * future
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step % 20 == 0 or step == 1:
                    state = dict(
                        status="action_training",
                        epoch=epoch,
                        total_epochs=args.epochs,
                        step=step,
                        prior_action_loss=float(prior.detach()),
                        repair_objective=repair,
                        batch_max_weight=float((w * quality).max()),
                        posterior_action_loss=float(posterior.detach()),
                        kl=float(prediction["kl"].detach()),
                        future_tool_mae_m=float(future.detach()) * 0.1,
                        gradient_norm=float(norm),
                        unique_windows_seen=len(seen),
                        train_windows=len(train.indices),
                        pid=os.getpid(),
                        elapsed_seconds=time.time() - began,
                        device=str(device),
                        cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0,
                        production_admission=False,
                    )
                    _atomic_json(out / "run_state.json", state)
                    with (out / "training_metrics.jsonl").open("a") as f:
                        f.write(json.dumps(state) + "\n")
                    print(json.dumps(state), flush=True)
            _atomic_json(
                out / "run_state.json",
                dict(
                    status="validating_prior",
                    epoch=epoch,
                    step=step,
                    pid=os.getpid(),
                    production_admission=False,
                ),
            )
            metrics = evaluate_prior(model, val, args.batch_size)
            save(out / "checkpoint_last.pt", epoch)
            if args.save_every_epoch:
                save(out / f"checkpoint_epoch_{epoch:03d}.pt", epoch)
            if metrics["selection_score"] < best:
                best = metrics["selection_score"]
                save(out / "checkpoint_best.pt", epoch)
            metrics.update(epoch=epoch, step=step, best_score=best)
            with (out / "validation_metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
        deltas = {n: float((p.detach() - initial[n]).abs().max()) for n, p in model.named_parameters()}
        _atomic_json(
            out / "parameter_update_audit.json",
            dict(
                max_abs_delta_by_parameter=deltas,
                main_action_parameters_changed=any(v > 0 for n, v in deltas.items() if "action_decoder" in n),
                language_parameters_changed=any(
                    v > 0 for n, v in deltas.items() if "language_embedding" in n
                ),
            ),
        )
        _atomic_json(
            out / "run_state.json",
            dict(
                status="training_complete_pending_closed_loop",
                steps=step,
                best_validation_score=best,
                initial_validation_score=baseline["selection_score"],
                unique_windows_seen=len(seen),
                train_windows=len(train.indices),
                window_coverage=len(seen) / len(train.indices),
                elapsed_seconds=time.time() - began,
                pid=os.getpid(),
                closed_loop_evaluated=False,
                production_admission=False,
                final_vla_acceptance=False,
            ),
        )
    except BaseException as exc:
        _atomic_json(
            out / "run_state.json",
            dict(
                status="failed",
                exception=repr(exc),
                pid=os.getpid(),
                elapsed_seconds=time.time() - began,
                production_admission=False,
            ),
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--validation-manifest", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--future-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=3101)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    p.add_argument("--cache-episodes", type=int, default=4)
    p.add_argument(
        "--initial-checkpoint", help="Continue ACT weights with a fresh optimizer; not exact run resumption"
    )
    p.add_argument("--unfreeze-rgb", action="store_true")
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--repair-objective", action="store_true", help="Run33: bounded normalized weights, first-step and demonstrated-hold emphasis")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.cache_episodes) < 1 or args.learning_rate <= 0:
        p.error("positive training settings required")
    run(args)


if __name__ == "__main__":
    main()
