"""Complete-pass integration pilot, not a production/admission trainer."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from .candidate_command_contract_v2 import ACTION_CONTRACT
from .temporal_input_contract_v3 import TemporalPackets, INPUT_KEYS
from .temporal_spatial_policy_v3 import TemporalSpatialPolicyV3
from .train_staged_hybrid_contact_sac import _atomic_json


def depth_loss(prediction, target, frame_mask, uncertainty=None):
    b, t, h, w = prediction.shape
    y = F.interpolate(target.reshape(b * t, 1, *target.shape[-2:]), size=(h, w), mode="nearest").reshape(
        b, t, h, w
    )
    valid = frame_mask[:, :, None, None] & (y > 0.01) & (y < 2)
    if not valid.any():
        return prediction.sum() * 0
    error = (prediction - y).abs()
    if uncertainty is None:
        return error[valid].mean()
    return (error / uncertainty + torch.log(uncertainty))[valid].mean()


def future_tool_loss(prediction, target, mask):
    if not mask.any():
        return prediction.sum() * 0
    return ((prediction - target).abs().mean(-1)[mask] / 0.1).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packets", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--depth-epochs", type=int, default=8)
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2901)
    rng = np.random.default_rng(2901)
    rows = json.loads((Path(a.packets) / "manifest.json").read_text())["records"]
    model = TemporalSpatialPolicyV3()
    cfg = model.config
    train = TemporalPackets([r for r in rows if r["split"] == "train"], cfg)
    val = TemporalPackets([r for r in rows if r["split"] == "validation"], cfg)
    plan = dict(
        config=asdict(cfg),
        parameters=sum(p.numel() for p in model.parameters()),
        train_episodes=len(train.records),
        validation_episodes=len(val.records),
        train_windows=len(train.indices),
        validation_windows=len(val.indices),
        epochs=a.epochs,
        depth_epochs=a.depth_epochs,
        input_keys=sorted(INPUT_KEYS),
        sampling="complete shuffled passes without replacement",
        action_contract=ACTION_CONTRACT,
        role="integration_pilot_not_formal_architecture_acceptance",
        production_admission=False,
        persistent_semantic_entity_map=False,
        multiview=False,
    )
    plan["future_motion_supervision"] = (
        "future FK tool XYZ auxiliary label only; never supplied as an input reference path"
    )
    plan["auxiliary_loss_weights"] = {"depth_nll": 0.02, "future_tool_mae_per_0p1m": 0.05}
    _atomic_json(out / "plan.json", plan)
    # Fail before spending training time: real packet starts and valid histories, every source/split.
    model.eval()
    preflight = []
    with torch.no_grad():
        for dataset in (train, val):
            for ri, record in enumerate(dataset.records):
                candidates = [i for i, (r, t) in enumerate(dataset.indices) if r == ri]
                for i in (candidates[0], candidates[len(candidates) // 2]):
                    batch = dataset.batch([i])
                    prediction = model(batch["inputs"])
                    if not torch.isfinite(prediction).all():
                        raise ValueError("nonfinite real-packet preflight")
                preflight.append(
                    dict(packet=record["packet"], source=record["source"], split=record["split"], passed=True)
                )
    _atomic_json(out / "preflight.json", dict(records=preflight, passed=True, production_admission=False))
    model.train()
    # Pretrain the RGB-only depth student; validation frames never enter updates.
    drows = []
    for r in train.records:
        with np.load(r["packet"]) as z:
            ids = np.flatnonzero(z["auxiliary_depth_frame_mask"])[::2]
            for i in ids:
                drows.append((z["rgb"][i], z["auxiliary_depth_mm"][i]))
    depth_optimizer = torch.optim.AdamW(model.depth_student.parameters(), lr=1e-3)
    began = time.time()
    for epoch in range(a.depth_epochs):
        totals = []
        for start in range(0, len(drows), 8):
            if start == 0:
                order = rng.permutation(len(drows))
            batch = [drows[i] for i in order[start : start + 8]]
            x = torch.from_numpy(np.stack([x[0] for x in batch])).float().permute(0, 3, 1, 2) / 255
            y = torch.from_numpy(np.stack([x[1] for x in batch])).float() / 1000
            d, u = model.depth_student(x)
            loss = depth_loss(d[:, None], y[:, None], torch.ones(len(batch), 1, dtype=torch.bool), u[:, None])
            depth_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.depth_student.parameters(), 1)
            depth_optimizer.step()
            totals.append(float(loss.detach()))
        state = dict(
            status="depth_pretraining",
            epoch=epoch + 1,
            total_epochs=a.depth_epochs,
            loss=float(np.mean(totals)),
            unique_train_depth_frames=len(drows),
            production_admission=False,
        )
        _atomic_json(out / "run_state.json", state)
        print(json.dumps(state), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    seen = set()
    best = float("inf")
    step = 0
    for epoch in range(a.epochs):
        order = rng.permutation(len(train.indices))
        model.train()
        for start in range(0, len(order), 2):
            ids = order[start : start + 2].tolist()
            seen.update(ids)
            b = train.batch(ids)
            pred = model(b["inputs"], return_aux=True)
            per = (pred["action"] - b["target"]).abs().mean(-1)
            weights = torch.tensor(
                [1 if train.records[train.indices[i][0]]["source"] == "rl" else 0.25 for i in ids]
            )
            loss = (((per * b["mask"]).sum(1) / b["mask"].sum(1)) + 0.5 * per[:, 0]).mul(weights).mean()
            aux = depth_loss(
                pred["estimated_depth_m"],
                b["auxiliary_depth_m"],
                b["auxiliary_depth_mask"],
                pred["depth_uncertainty_m"],
            )
            motion = future_tool_loss(pred["next_tool_xyz"], b["future_tool_xyz"], b["future_tool_mask"])
            total = loss + 0.02 * aux + 0.05 * motion
            if not torch.isfinite(total):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            step += 1
            if step % 20 == 0:
                state = dict(
                    status="action_integration_training",
                    epoch=epoch + 1,
                    total_epochs=a.epochs,
                    step=step,
                    action_loss=float(loss.detach()),
                    depth_aux_nll=float(aux.detach()),
                    future_tool_mae_m=float(motion.detach()) * 0.1,
                    unique_windows_seen=len(seen),
                    train_windows=len(train.indices),
                    elapsed_seconds=time.time() - began,
                    pid=os.getpid(),
                    production_admission=False,
                )
                _atomic_json(out / "run_state.json", state)
                print(json.dumps(state), flush=True)
        model.eval()
        errors = idle = elements = depth_total = depth_count = 0
        with torch.no_grad():
            for start in range(0, len(val.indices), 2):
                b = val.batch(list(range(start, min(start + 2, len(val.indices)))))
                p = model(b["inputs"], return_aux=True)
                mask = b["mask"][..., None].expand_as(p["action"])
                errors += float((p["action"] - b["target"]).abs()[mask].sum())
                idle += float(b["target"].abs()[mask].sum())
                elements += int(mask.sum())
                if b["auxiliary_depth_mask"].any():
                    depth_total += float(
                        depth_loss(p["estimated_depth_m"], b["auxiliary_depth_m"], b["auxiliary_depth_mask"])
                    )
                    depth_count += 1
        score = errors / elements
        checkpoint = dict(
            model=model.state_dict(),
            config=asdict(cfg),
            action_contract=ACTION_CONTRACT,
            input_keys=sorted(INPUT_KEYS),
            format="temporal_spatial_v3_integration",
            epoch=epoch + 1,
            production_admission=False,
        )
        if score < best:
            best = score
            torch.save(checkpoint, out / "checkpoint_best.pt")
        torch.save(checkpoint, out / "checkpoint_last.pt")
        result = dict(
            status="complete" if epoch + 1 == a.epochs else "epoch_complete",
            epoch=epoch + 1,
            total_epochs=a.epochs,
            steps=step,
            validation_command_mae=score,
            zero_command_mae=idle / elements,
            validation_depth_batch_mean_mae_m=depth_total / max(depth_count, 1),
            unique_windows_seen=len(seen),
            train_windows=len(train.indices),
            coverage_fraction=len(seen) / len(train.indices),
            closed_loop_evaluated=False,
            spatial_accuracy_accepted=False,
            production_admission=False,
        )
        _atomic_json(out / "run_state.json", result)
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
