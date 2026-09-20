"""Export raw production shards to an official local LeRobotDataset v3 layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets import LeRobotDataset

from .config import ARTIFACTS_DIR


def features(height: int, width: int) -> dict:
    image = {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    depth = {
        "dtype": "video",
        "shape": (height, width, 1),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": True},
    }
    return {
        "observation.images.wrist": image,
        "observation.images.wrist_depth": depth,
        "observation.images.wrist_segmentation": image,
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": [*[f"{name}.pos" for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")], *[f"{name}.vel" for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]],
        },
        "observation.tool_pose": {"dtype": "float32", "shape": (12,), "names": None},
        "observation.task_vector": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["target_x", "target_y", "obstacle_x", "obstacle_y", "obstacle_enabled", "teacher_confidence"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"{name}.delta" for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")],
        },
        "action.joint_target": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"{name}.target" for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
        "next.success": {"dtype": "bool", "shape": (1,), "names": ["success"]},
        "teacher.phase": {"dtype": "int64", "shape": (1,), "names": ["phase_id"]},
    }


SEGMENTATION_PALETTE = np.asarray(
    [[0, 0, 0], [255, 210, 20], [30, 90, 240], [40, 210, 90], [220, 40, 30], [240, 240, 220], [155, 155, 145]],
    dtype=np.uint8,
)


def _align_episode_video_timestamps(output: Path) -> dict:
    """Align episode offsets with the PTS of the finalized video files.

    LeRobot 0.5.2 derives each episode offset by accumulating the duration of
    the temporary per-episode files.  For 12-bit HEVC depth video, those
    durations are quantized slightly differently from the PTS in the final
    concatenated MP4.  The error accumulates over episodes and can make the
    standard strict LeRobot reader reject otherwise valid frames.  Rebuilding
    the offsets from the finalized files keeps the v3 metadata synchronized
    with what the decoder actually sees.
    """
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    episode_paths = sorted((output / "meta" / "episodes").rglob("*.parquet"))
    tables = [pq.read_table(path) for path in episode_paths]
    timestamp_columns = {
        key: (f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp") for key in video_keys
    }
    updates = [
        {
            column: table[column].to_pylist()
            for columns in timestamp_columns.values()
            for column in columns
        }
        for table in tables
    ]
    max_correction = 0.0
    video_files = 0

    for video_key in video_keys:
        groups: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        chunk_column = f"videos/{video_key}/chunk_index"
        file_column = f"videos/{video_key}/file_index"
        for table_index, table in enumerate(tables):
            for row_index in range(table.num_rows):
                group_key = (
                    int(table[chunk_column][row_index].as_py()),
                    int(table[file_column][row_index].as_py()),
                )
                groups.setdefault(group_key, []).append(
                    (
                        int(table["episode_index"][row_index].as_py()),
                        table_index,
                        row_index,
                        int(table["length"][row_index].as_py()),
                    )
                )

        from_column, to_column = timestamp_columns[video_key]
        for (chunk_index, file_index), episodes in groups.items():
            video_path = output / info["video_path"].format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                frame_timestamps = [
                    float(frame.pts * stream.time_base)
                    for frame in container.decode(stream)
                    if frame.pts is not None
                ]
            episodes.sort(key=lambda item: item[0])
            expected_frames = sum(item[3] for item in episodes)
            if len(frame_timestamps) != expected_frames:
                raise RuntimeError(
                    f"Video frame count mismatch for {video_key} {video_path}: "
                    f"decoded={len(frame_timestamps)} expected={expected_frames}"
                )
            if any(current <= previous for previous, current in zip(frame_timestamps, frame_timestamps[1:])):
                raise RuntimeError(f"Non-monotonic video timestamps in {video_path}")

            frame_offset = 0
            for _, table_index, row_index, length in episodes:
                aligned_from = frame_timestamps[frame_offset]
                next_offset = frame_offset + length
                aligned_to = (
                    frame_timestamps[next_offset]
                    if next_offset < len(frame_timestamps)
                    else frame_timestamps[-1] + 1.0 / fps
                )
                max_correction = max(
                    max_correction,
                    abs(updates[table_index][from_column][row_index] - aligned_from),
                    abs(updates[table_index][to_column][row_index] - aligned_to),
                )
                updates[table_index][from_column][row_index] = aligned_from
                updates[table_index][to_column][row_index] = aligned_to
                frame_offset = next_offset
            video_files += 1

    for path, table, table_updates in zip(episode_paths, tables, updates):
        aligned = table
        for column, values in table_updates.items():
            column_index = aligned.schema.get_field_index(column)
            aligned = aligned.set_column(
                column_index,
                column,
                pa.array(values, type=aligned.schema.field(column_index).type),
            )
        temporary_path = path.with_suffix(".aligned.parquet")
        pq.write_table(aligned, temporary_path)
        temporary_path.replace(path)

    return {
        "video_files": video_files,
        "video_keys": video_keys,
        "max_timestamp_correction_seconds": max_correction,
    }


def export(
    raw_root: Path,
    split: str,
    output: Path,
    repo_id: str,
    successes_only: bool = True,
    max_episodes: int | None = None,
) -> dict:
    manifest_path = raw_root / split / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    selected = [entry for entry in entries if entry["success"] or not successes_only]
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive when provided")
        selected = selected[:max_episodes]
    if not selected:
        raise ValueError("No episodes selected for LeRobot export")
    first_path = raw_root / split / selected[0]["shard"]
    with h5py.File(first_path, "r") as stream:
        group = stream[selected[0]["group"]]
        height, width = group["rgb_wrist"].shape[1:3]
        fps = int(group.attrs["fps"])
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=fps,
        robot_type="so101_follower_edgearm",
        features=features(height, width),
        use_videos=True,
        image_writer_threads=8,
        batch_encoding_size=1,
        encoder_threads=4,
        data_files_size_in_mb=256,
        video_files_size_in_mb=512,
    )
    open_files: dict[Path, h5py.File] = {}
    action_contracts: set[str] = set()
    teacher_types: set[str] = set()
    teacher_checkpoint_sha256s: set[str] = set()
    try:
        for episode_number, entry in enumerate(selected):
            shard_path = raw_root / split / entry["shard"]
            stream = open_files.setdefault(shard_path, h5py.File(shard_path, "r"))
            group = stream[entry["group"]]
            action_contracts.add(str(group.attrs.get("action_target_contract", "legacy_raw_request")))
            teacher_types.add(str(group.attrs.get("teacher_type", "production_geometric")))
            teacher_checkpoint_sha256s.add(str(group.attrs.get("teacher_checkpoint_sha256", "")))
            task = str(group.attrs["task"])
            obstacle_enabled = float(bool(group.attrs["obstacle"]))
            length = len(group["action_joint_delta"])
            for frame_index in range(length):
                wrist_seg = SEGMENTATION_PALETTE[np.clip(group["segmentation_wrist"][frame_index], 0, 6)]
                state = np.concatenate(
                    [group["joint_position"][frame_index], group["joint_velocity"][frame_index]]
                ).astype(np.float32)
                task_vector = np.concatenate(
                    [
                        group["target_xy"][frame_index],
                        group["obstacle_xy"][frame_index],
                        [obstacle_enabled, group["teacher_confidence"][frame_index]],
                    ]
                ).astype(np.float32)
                dataset.add_frame(
                    {
                        "task": task,
                        "observation.images.wrist": group["rgb_wrist"][frame_index],
                        "observation.images.wrist_depth": group["depth_wrist_mm"][frame_index][..., None],
                        "observation.images.wrist_segmentation": wrist_seg,
                        "observation.state": state,
                        "observation.tool_pose": group["tool_pose"][frame_index].astype(np.float32),
                        "observation.task_vector": task_vector,
                        "action": group["action_joint_delta"][frame_index].astype(np.float32),
                        "action.joint_target": group["action_joint_target"][frame_index].astype(np.float32),
                        "next.reward": np.asarray([group["reward"][frame_index]], dtype=np.float32),
                        "next.done": np.asarray([group["done"][frame_index]], dtype=bool),
                        "next.success": np.asarray([group["success"][frame_index]], dtype=bool),
                        "teacher.phase": np.asarray([group["phase_id"][frame_index]], dtype=np.int64),
                    }
                )
            dataset.save_episode(parallel_encoding=False)
            if (episode_number + 1) % 10 == 0 or episode_number + 1 == len(selected):
                print(f"exported={episode_number + 1}/{len(selected)}", flush=True)
        dataset.finalize()
    finally:
        for stream in open_files.values():
            stream.close()
    timestamp_alignment = _align_episode_video_timestamps(output)
    return {
        "raw_episodes": len(entries),
        "exported_episodes": len(selected),
        "output": str(output),
        "repo_id": repo_id,
        "successes_only": successes_only,
        "max_episodes": max_episodes,
        "action_target_contracts": sorted(action_contracts),
        "teacher_types": sorted(teacher_types),
        "teacher_checkpoint_sha256s": sorted(value for value in teacher_checkpoint_sha256s if value),
        "timestamp_alignment": timestamp_alignment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=ARTIFACTS_DIR / "production_wrist_rl_dataset_v1"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "lerobot_v3" / "edgearm_wrist_rl_production_v1",
    )
    parser.add_argument("--repo-id", default="local/edgearm-wrist-rl-production-v1")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.raw_root,
                args.split,
                args.output,
                args.repo_id,
                not args.include_failures,
                args.max_episodes,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
