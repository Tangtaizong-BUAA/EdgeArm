"""Repeat deterministic residual evaluation before interpreting learning curves."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path
import numpy as np
import torch
from .act_residual_ppo_run33 import ResidualACT
from .evaluate_multimodal_act_v5 import load_model, rollout
from .train_multimodal_act_v5 import sha256
from .train_staged_hybrid_contact_sac import _atomic_json


def deterministic_runtime():
    torch.set_num_threads(4)
    torch.manual_seed(3401)
    np.random.seed(3401)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--residual", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=12000010)
    a = p.parse_args()
    deterministic_runtime()
    base, checkpoint = load_model(a.checkpoint)
    saved = torch.load(a.residual, map_location="cpu", weights_only=True)
    if saved["base_checkpoint_sha256"] != sha256(a.checkpoint):
        raise ValueError("base mismatch")
    model = ResidualACT(base, saved["limit"]).cuda().eval()
    model.head.load_state_dict(saved["head"])
    model.explore = False
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for i in range(2):
        folder = output / f"repeat_{i}"
        folder.mkdir()
        model.buffer.clear()
        result = rollout(model, checkpoint, a.seed, folder, 900, "task_goal_v1", record_video=False)
        results.append(result)
        _atomic_json(folder / "result.json", result)
    traces = [
        np.load(output / f"repeat_{i}" / f"{a.seed}_trace.npz")["reported_command_next_reported"]
        for i in range(2)
    ]
    equal = traces[0].shape == traces[1].shape and np.array_equal(*traces)
    _atomic_json(
        output / "summary.json", dict(results=results, exact_trace_repeat=equal, production_admission=False)
    )
    print(
        {
            "exact_trace_repeat": equal,
            "results": [
                {k: r[k] for k in ("safe_success", "maximum_coverage", "maximum_hold_s")} for r in results
            ],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
