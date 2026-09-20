"""Bounded keypoint training on authorized collection only, no policy tuning."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .run34_repeat_eval import deterministic_runtime
from .run82_train_spatial import Sequences
from .run88_keypoint_model import WristKeypoints, keypoint_loss, project_labels
from .train_staged_hybrid_contact_sac import _atomic_json


def load_frames(dataset, publish):
    rows = {k: [] for k in ("rgb", "pose", "K", "xyz", "present", "visible", "validation", "route")}
    for index, row in enumerate(dataset.records):
        if row["split"] == "validation" and row["variant"] != -1:
            continue
        ins, labs = dataset.load(row)
        n = len(ins["rgb"])
        rows["rgb"].append(ins["rgb"])
        rows["pose"].append(ins["pose"])
        rows["K"].append(np.broadcast_to(ins["K"], (n, 3, 3)))
        for key in ("xyz", "present", "visible"):
            rows[key].append(labs[key])
        rows["validation"].append(np.full(n, row["split"] == "validation", bool))
        rows["route"].append(np.full(n, row["route"], np.int64))
        if index % 24 == 0:
            publish(index + 1, len(dataset.records))
    return {k: torch.from_numpy(np.concatenate(v)).cuda() for k, v in rows.items()}


def validate(model, data):
    ids = torch.nonzero(data["validation"]).flatten()
    errors = []
    selected_errors = []
    missing = accepted_missing = visible_count = accepted_visible = 0
    model.eval()
    with torch.inference_mode():
        for chunk in ids.split(128):
            out = model(data["rgb"][chunk], data["pose"][chunk], data["K"][chunk])
            _, inside = project_labels(data["xyz"][chunk], data["pose"][chunk], data["K"][chunk])
            known = data["present"][chunk] & data["visible"][chunk] & inside
            accepted = (out["confidence"] >= 0.95) & out["valid"]
            err = (out["xyz"][..., :2] - data["xyz"][chunk][..., :2]).norm(dim=-1) * 1000
            errors.append(err[known].cpu())
            selected_errors.append(err[known & accepted].cpu())
            missing += int((~known).sum())
            accepted_missing += int((accepted & ~known).sum())
            visible_count += int(known.sum())
            accepted_visible += int((accepted & known).sum())
    all_error = torch.cat(errors)
    qualified = torch.cat(selected_errors)

    def stats(x):
        return (
            dict(
                count=len(x),
                mean_mm=float(x.mean()),
                median_mm=float(x.median()),
                p90_mm=float(x.quantile(0.9)),
            )
            if len(x)
            else dict(count=0)
        )

    return dict(
        all_visible=stats(all_error),
        accepted_visible=stats(qualified),
        visible_recall=accepted_visible / max(visible_count, 1),
        false_acceptance=accepted_missing / max(missing, 1),
        confidence_threshold=0.95,
        old_sequence_validation_not_independent=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=1800)
    p.add_argument("--eval-every", type=int, default=600)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    if not 4 <= a.updates <= 3600:
        raise ValueError("bounded keypoint experiment")
    a.output.mkdir(parents=True, exist_ok=False)
    deterministic_runtime()
    torch.set_num_threads(2)
    state = dict(
        run="Run88",
        status="running",
        phase="prepare_keypoint_frames",
        step=0,
        total_updates=a.updates,
        started=time.time(),
        evaluations=[],
        training_kind="image_heatmap_and_calibrated_ray_supervision",
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        actor_uses_simulator_state=False,
        nominal_height_prior=True,
        learned_depth=False,
        action_policy_changed=False,
        independent_acceptance=False,
        production_admission=False,
        export_admission=False,
        final_vla_acceptance=False,
    )
    source = a.output / "source_snapshot"
    source.mkdir()
    for name in ("run88_train_keypoints.py", "run88_keypoint_model.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)

    def publish():
        state.update(updated=time.time(), elapsed_seconds=time.time() - state["started"])
        _atomic_json(a.output / "run_state.json", state)

    def progress(n, total):
        state.update(recode_completed=n, recode_total=total)
        publish()

    publish()
    try:
        dataset = Sequences(a.manifest)
        if a.smoke_only:
            dataset.records = [r for r in dataset.records if r["split"] == "train" and r["variant"] == -1][
                :9
            ] + [r for r in dataset.records if r["split"] == "validation" and r["variant"] == -1][:2]
        data = load_frames(dataset, progress)
        model = WristKeypoints().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True)
        pools = [torch.nonzero((~data["validation"]) & (data["route"] == r)).flatten() for r in range(9)]
        pools = [x for x in pools if len(x)]
        state.update(
            training_frames=int((~data["validation"]).sum()),
            validation_frames=int(data["validation"].sum()),
            parameters=sum(x.numel() for x in model.parameters()),
            phase="keypoint_training",
        )
        best = None
        for step in range(1, a.updates + 1):
            ids = torch.cat([pool[torch.randint(len(pool), (16,), device="cuda")] for pool in pools])
            model.train()
            out = model(data["rgb"][ids], data["pose"][ids], data["K"][ids])
            loss, metrics = keypoint_loss(
                out,
                data["xyz"][ids],
                data["present"][ids],
                data["visible"][ids],
                data["pose"][ids],
                data["K"][ids],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            state.update(step=step, metrics=dict(loss=float(loss.detach()), **metrics))
            if step % 20 == 0:
                publish()
                with (a.output / "training_metrics.jsonl").open("a") as f:
                    f.write(json.dumps(dict(step=step, metrics=state["metrics"])) + "\n")
            if step % a.eval_every == 0 or step == a.updates:
                summary = validate(model, data) | dict(step=step)
                path = a.output / f"keypoints_step_{step}.pt"
                torch.save(
                    dict(
                        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        policy_kind=model.kind,
                        confidence_threshold=0.95,
                        nominal_height_prior=True,
                        actor_uses_simulator_state=False,
                        validation=summary,
                        production_admission=False,
                    ),
                    path,
                )
                quality = summary["accepted_visible"]
                eligible = (
                    quality["count"] >= 500
                    and quality.get("median_mm", 999) <= 8
                    and quality.get("p90_mm", 999) <= 25
                    and summary["false_acceptance"] <= 0.02
                )
                rank = (eligible, -quality.get("p90_mm", 999), summary["visible_recall"])
                if best is None or rank > best:
                    best = rank
                    state.update(best_checkpoint=str(path), best_validation_rank=list(rank))
                state["perception_gate_passed"] = bool(best[0])
                state["evaluations"].append(summary)
                publish()
        state.update(status="smoke_complete" if a.smoke_only else "complete_pending_review", phase="finished")
        publish()
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        publish()
        raise


if __name__ == "__main__":
    main()
