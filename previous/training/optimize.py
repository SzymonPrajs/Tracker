#!/usr/bin/env python3
"""Restartable successive-halving search for the room-head detector."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "training" / "train.py"
EVALUATE = ROOT / "training" / "evaluate.py"
DEFAULT_ROOT = ROOT / "training" / "runs" / "optimization"
FINAL_EPOCHS = 30

CONFIGURATIONS = {
    "baseline": [],
    "baseline_seed19": ["--seed", "19"],
    "lr_low": ["--lr", "0.0015"],
    "lr_high": ["--lr", "0.005"],
    "size_strong": ["--size-weight", "0.5"],
    "light_augmentation": [
        "--brightness", "0.04", "--contrast", "0.06", "--channel-jitter", "0.03",
    ],
    "strong_augmentation": [
        "--brightness", "0.15", "--contrast", "0.20", "--channel-jitter", "0.10",
    ],
    "cosine": ["--scheduler", "cosine"],
    "sgd": ["--optimizer", "sgd", "--lr", "0.03", "--weight-decay", "0.0001"],
}


def environment() -> dict[str, str]:
    result = dict(os.environ)
    training = str(ROOT / "training")
    result["PYTHONPATH"] = training + os.pathsep + result.get("PYTHONPATH", "")
    return result


def completed_epochs(directory: Path) -> int:
    history = directory / "history.jsonl"
    if not history.exists():
        return 0
    epochs = [json.loads(line)["epoch"] for line in history.read_text().splitlines() if line.strip()]
    return max(epochs, default=0)


def train_candidate(name: str, arguments: list[str], budget: int, root: Path) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    completed = completed_epochs(directory)
    if completed >= budget:
        print(f"[{name}] already has {completed} epochs; target is {budget}", flush=True)
        return
    command = [
        sys.executable, str(TRAIN),
        "--output", str(directory),
        "--epochs", str(budget),
        "--schedule-epochs", str(FINAL_EPOCHS),
        "--patience", "0",
        "--save-every-epoch",
        "--log-interval", "200",
        *arguments,
    ]
    checkpoint = directory / "last.pt"
    if completed and checkpoint.exists():
        command.extend(("--resume", str(checkpoint)))
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment(), check=True)


def evaluate_checkpoint(checkpoint: Path, metrics: Path) -> dict:
    if not metrics.exists():
        command = [
            sys.executable, str(EVALUATE), str(checkpoint),
            "--split", "val", "--output", str(metrics),
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment(), check=True)
    return json.loads(metrics.read_text())


def best_candidate_metric(name: str, budget: int, root: Path) -> dict:
    directory = root / name
    checkpoints = sorted(
        checkpoint for checkpoint in directory.glob("epoch-*.pt")
        if int(checkpoint.stem.split("-")[-1]) <= budget
    )
    if not checkpoints:
        checkpoints = [directory / "model.pt"]
    best = None
    for checkpoint in checkpoints:
        label = checkpoint.stem
        metrics = evaluate_checkpoint(checkpoint, directory / f"metrics-{label}.json")
        row = {
            "candidate": name,
            "budget": budget,
            "checkpoint": str(checkpoint),
            **metrics,
        }
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    return best


def write_leaderboard(root: Path, stage: int, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    payload = {"stage_budget": stage, "ranking": ordered}
    (root / "leaderboard.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nLeaderboard after {stage} epochs", flush=True)
    for rank, row in enumerate(ordered, 1):
        print(
            f"{rank:2d}. {row['candidate']:22} score={row['selection_score']:.4f} "
            f"room_mAP={row['room_map30_50']:.4f} all_mAP={row['map30_50']:.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    candidates = list(CONFIGURATIONS)
    stages = ((4, 3), (10, 1), (FINAL_EPOCHS, 1))
    winner = None
    for budget, promote in stages:
        rows = []
        print(f"\n=== stage: train {len(candidates)} candidate(s) through epoch {budget} ===", flush=True)
        for name in candidates:
            train_candidate(name, CONFIGURATIONS[name], budget, args.root)
            row = best_candidate_metric(name, budget, args.root)
            rows.append(row)
            write_leaderboard(args.root, budget, rows)
        rows.sort(key=lambda row: row["selection_score"], reverse=True)
        candidates = [row["candidate"] for row in rows[:promote]]
        winner = rows[0]
        print(f"promoting: {', '.join(candidates)}", flush=True)

    assert winner is not None
    winning_checkpoint = Path(winner["checkpoint"])
    best_model = args.root / "best_model.pt"
    shutil.copy2(winning_checkpoint, best_model)
    artifacts_model = ROOT / "training" / "artifacts" / "optimized_model.pt"
    artifacts_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winning_checkpoint, artifacts_model)
    test_metrics = args.root / "test_metrics.json"
    command = [
        sys.executable, str(EVALUATE), str(best_model),
        "--split", "test", "--output", str(test_metrics),
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment(), check=True)
    summary = {
        "winner": winner,
        "best_model": str(best_model),
        "artifacts_model": str(artifacts_model),
        "test": json.loads(test_metrics.read_text()),
    }
    (args.root / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
