"""Persistent simulator workers with causal RGB and complete replay records.

No hardware I/O. A fragment is not a reset or terminal. Evaluation sessions
are isolated from the paused training plant and never enter replay.
"""

import atexit
from copy import deepcopy
import os
from pathlib import Path
import time

import numpy as np
import torch

from .branch_session_run37 import BranchSession
from .continuous_recovery_run38 import ContinuousActor, FrozenReadout, FIELDS
from .evaluate_multimodal_act_v5 import load_model
from .recovery_core_run37 import shaped_rewards
from .run34_repeat_eval import deterministic_runtime
from .train_multimodal_act_v5 import sha256
from .train_staged_hybrid_contact_sac import _atomic_json

WORKER = None


def privileged_state(session):
    """Training critic only; nominal plant has zero delay and fixed dynamics."""
    env = session.env
    if env._command_delay_steps != 0 or len(env._command_queue):
        raise ValueError("Run38 nominal critic contract requires empty actuator queue")
    anchor = env._workspace_recovery_anchor_v10
    anchor = np.zeros(6) if anchor is None else np.asarray(anchor)
    values = np.concatenate((
        env.data.qpos, env.data.qvel, env.data.act, env.data.ctrl,
        env._servo_velocity, env._backlash_remaining, env._last_motor_direction,
        env._last_actual_velocity, env._last_actuator_force, env._motor_temperature_c / 100,
        anchor, env.target_xy, session.reward_state(),
        [max(env.config.max_steps - env.step_count, 0) / env.config.max_steps,
         env._loaded_voltage_v / 12, env._runtime_pusher_desk_safety_stop_requested],
    )).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite privileged critic state")
    return values


class EpisodeRun:
    def __init__(self, readout, checkpoint, seed, path, source, sampling_seed):
        self.readout = readout
        self.session = BranchSession(readout.config, checkpoint["action_contract"], int(seed), "cuda")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.rng = np.random.default_rng(sampling_seed)
        self.source, self.seed = source, int(seed)
        self.rows, self.telemetry = [], []
        self.initial = self.session.reward_state()
        self.max_cov, self.max_hold = self.initial[1:]
        self.contacts = self.effectful = self.rewrites = self.rejections = 0
        self.controlled = self.prefix_steps = 0
        self.fragment_id = 0
        self.started = time.monotonic()
        self.cached = None
        self.closed = False

    def features(self):
        if self.cached is None:
            inputs = self.session.observe()
            with torch.inference_mode():
                x, base = self.readout(inputs)
            self.cached = (x[0].cpu().numpy().copy(), base[0].cpu().numpy().copy(), privileged_state(self.session))
        return self.cached

    def advance(self, action, *, learn):
        before = self.features()
        state_before = self.session.reward_state()
        result = self.session.advance(np.asarray(action, np.float32))
        # Save actual row including completed command/feedback transaction.
        self.rows.append(deepcopy(self.session.buffer.rows[-1]))
        after_state = self.session.reward_state()
        self.cached = None
        after = self.features()
        reward, _ = shaped_rewards(np.stack((state_before, after_state)), end_kind=self.session.end_kind)
        terminal = self.session.end_kind != "sampler_cut"
        self.max_cov = max(self.max_cov, after_state[1])
        self.max_hold = max(self.max_hold, after_state[2])
        self.contacts += result.get("valid_contact", False)
        self.effectful += result.get("effectful_contact", False)
        self.rewrites += result.get("safety_rewrite", False)
        self.rejections += result["rejected"]
        self.controlled += int(learn)
        self.prefix_steps += int(not learn)
        self.telemetry.append([self.session.env.data.time, *after_state, float(learn),
                               float(result.get("valid_contact", False)), float(result.get("effectful_contact", False)),
                               float(result.get("safety_rewrite", False)), float(result["rejected"]), float(reward[0])])
        return dict(x=before[0], base=before[1], privileged=before[2], action=np.asarray(action, np.float32),
                    reward=reward[0], next_x=after[0], next_base=after[1], next_privileged=after[2],
                    terminal=float(terminal))

    def rollin(self, mode, packet):
        if mode == "normal":
            return
        with np.load(packet) as z:
            commands = z["command"].copy()
        threshold = float(self.rng.uniform(0.80, 0.96))
        limit = min(len(commands), 240) if mode == "near" else int(self.rng.integers(135, 176))
        for t in range(limit):
            if self.session.end_kind != "sampler_cut":
                break
            self.features()
            action = commands[t] if mode == "near" else self.cached[1]
            self.advance(action, learn=False)
            if mode == "near" and self.session.reward_state()[1] >= threshold:
                break
        if mode == "recovery" and self.session.end_kind == "sampler_cut":
            # Real premature-stop history; no object or joint teleportation.
            for _ in range(int(self.rng.integers(8, 25))):
                self.advance(np.zeros(6, np.float32), learn=False)
                if self.session.end_kind != "sampler_cut":
                    break

    def summary(self):
        return dict(seed=self.seed, source=self.source, steps=len(self.rows), actor_steps=self.controlled,
                    training_rollin_steps=self.prefix_steps, end_kind=self.session.end_kind, reason=self.session.reason,
                    safe_success=self.session.end_kind == "success" and self.controlled > 0,
                    prefix_only_success=self.session.end_kind == "success" and self.controlled == 0,
                    initial_coverage=float(self.initial[1]), maximum_coverage=float(self.max_cov),
                    maximum_hold_s=float(self.max_hold), final_reward_state=self.session.reward_state().tolist(),
                    valid_contact_steps=int(self.contacts), effectful_contact_steps=int(self.effectful),
                    safety_rewrite_steps=int(self.rewrites), rejected_commands=int(self.rejections),
                    elapsed_seconds=time.monotonic() - self.started,
                    instruction=self.session.buffer.instruction, start_stage="CONTACT_TRANSPORT_HOLD",
                    exact_home_evaluated=False, production_admission=False, export_admission=False, final_vla_acceptance=False)

    def persist(self, final=False):
        _atomic_json(self.path / ("result.json" if final else "episode_progress.json"), self.summary())
        if final:
            rows = self.rows + [deepcopy(self.session.buffer.rows[-1])]
            data = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
            data["K"] = self.session.buffer.K
            data["evaluation_telemetry"] = np.asarray(self.telemetry)
            np.savez_compressed(self.path / "wrist_command_episode.npz", **data)

    def close(self):
        if not self.closed:
            self.session.close()
            self.closed = True


def initialize(checkpoint_path, anchor_path, barrier):
    global WORKER
    deterministic_runtime()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base, checkpoint = load_model(checkpoint_path)
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=True)
    if anchor["base_checkpoint_sha256"] != sha256(checkpoint_path):
        raise ValueError("anchor lineage mismatch")
    readout = FrozenReadout(base, anchor).cuda().eval()
    WORKER = dict(readout=readout, checkpoint=checkpoint, actor=ContinuousActor(readout.feature_dim).cuda().eval(),
                  live=None, episode_number=0, sampling_initialized=False)
    atexit.register(close_live)
    barrier.wait(timeout=240)


def ready():
    return os.getpid()


def close_live():
    if WORKER and WORKER["live"] is not None:
        WORKER["live"].persist(final=True)
        WORKER["live"].close()
        WORKER["live"] = None


def tensor_batch(rows):
    return {k: np.asarray([r[k] for r in rows], np.float32) for k in FIELDS}


def isolated_episode(job):
    """Offline seed replay or eval; nested task_goal audit restores outer scope."""
    w = WORKER
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    torch.manual_seed(job["sampling_seed"])
    runner = EpisodeRun(w["readout"], w["checkpoint"], job["seed"], job["output"], job["source"], job["sampling_seed"])
    transitions = []
    try:
        commands = None
        if job["source"] == "offline_teacher_replay":
            with np.load(job["packet"]) as z:
                commands = z["command"].copy()
        else:
            w["actor"].load_state_dict(job["actor"])
        for t in range(min(job.get("max_steps", 900), len(commands) if commands is not None else 900)):
            x, base, _ = runner.features()
            if commands is not None:
                action = commands[t]
            else:
                with torch.inference_mode():
                    action = w["actor"](torch.from_numpy(x[None]).cuda(), torch.from_numpy(base[None]).cuda(),
                                         deterministic=True)[0][0].cpu().numpy()
            transitions.append(runner.advance(action, learn=True))
            if runner.session.end_kind != "sampler_cut":
                break
            if t % 100 == 99:
                runner.persist()
        runner.persist(final=True)
        batch = tensor_batch(transitions)
        if commands is not None:
            np.savez_compressed(runner.path / "transitions.npz", **batch)
        return dict(result=runner.summary(), episode_id=str(runner.path),
                    batch=batch if commands is not None else None)
    finally:
        runner.close()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)


def collect_fragment(job):
    w = WORKER
    if not w["sampling_initialized"]:
        torch.manual_seed(job["sampling_seed"])
        w["sampling_initialized"] = True
    w["actor"].load_state_dict(job["actor"])
    completed, chunks = [], []
    count = prefix_count = 0
    while count < job["steps"]:
        if w["live"] is None:
            number = w["episode_number"]
            record = job["records"][(job["slot"] + number * job["workers"]) % len(job["records"])]
            mode = ("normal", "normal", "near", "recovery")[(job["slot"] + number) % 4]
            path = Path(job["output"]) / f"worker_{os.getpid()}" / f"episode_{number:04d}_{mode}"
            runner = EpisodeRun(w["readout"], w["checkpoint"], record["seed"], path, mode,
                                job["sampling_seed"] + number)
            w["live"] = runner
            w["episode_number"] += 1
            runner.rollin(mode, record["packet"])
            prefix_count += runner.prefix_steps
            if runner.session.end_kind != "sampler_cut":
                runner.persist(final=True)
                completed.append(runner.summary())
                runner.close()
                w["live"] = None
                continue
        runner = w["live"]
        rows = []
        while count < job["steps"] and runner.session.end_kind == "sampler_cut":
            x, base, _ = runner.features()
            with torch.inference_mode():
                action = w["actor"](torch.from_numpy(x[None]).cuda(), torch.from_numpy(base[None]).cuda())[0][0].cpu().numpy()
            rows.append(runner.advance(action, learn=True))
            count += 1
        batch = tensor_batch(rows)
        path = runner.path / f"replay_{runner.fragment_id:04d}.npz"
        np.savez_compressed(path, **batch)
        chunks.append(dict(episode_id=str(runner.path), batch=batch))
        runner.fragment_id += 1
        runner.persist()
        if runner.session.end_kind != "sampler_cut":
            runner.persist(final=True)
            completed.append(runner.summary())
            runner.close()
            w["live"] = None
        else:
            # Persist RGB/command data at every fragment, including a live
            # episode, so interruption does not leave only latent replay.
            runner.persist(final=True)
    return dict(chunks=chunks, completed=completed, controls=count, prefix_controls=prefix_count,
                active=None if w["live"] is None else w["live"].summary(), policy_version=job["policy_version"])
