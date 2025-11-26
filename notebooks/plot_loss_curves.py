"""Plot training and evaluation loss trajectories for Waray and Cebuano experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def load_trainer_state(checkpoint_dir: Path) -> Dict:
    """Load the trainer_state.json file from a checkpoint directory."""
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing trainer_state.json in {checkpoint_dir}")
    with state_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def gather_logs(stage_label: str, stage_root: Path) -> pd.DataFrame:
    """Collect train/eval loss logs for a training stage into a DataFrame."""
    checkpoints = sorted(stage_root.glob("checkpoint-*/trainer_state.json"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {stage_root}")

    # Use the highest-step checkpoint for the most complete log history
    trainer_state = load_trainer_state(checkpoints[-1].parent)

    records: List[Dict] = []
    for entry in trainer_state.get("log_history", []):
        step = entry.get("step")
        epoch = entry.get("epoch")
        if step is None or epoch is None:
            continue

        if "loss" in entry:
            records.append(
                {
                    "stage": stage_label,
                    "metric": "train_loss",
                    "step": step,
                    "epoch": epoch,
                    "value": entry["loss"],
                }
            )

        if "eval_loss" in entry:
            records.append(
                {
                    "stage": stage_label,
                    "metric": "eval_loss",
                    "step": step,
                    "epoch": epoch,
                    "value": entry["eval_loss"],
                }
            )

    if not records:
        raise ValueError(f"No log records found for stage {stage_label}")

    return pd.DataFrame(records)


def build_dataset(repo_root: Path, language: str) -> pd.DataFrame:
    """Assemble log data for baseline and experimental stage 2."""
    stage_specs = [
        ("Baseline", repo_root / "models" / f"{language}_baseline_nllb_lora_bf16"),
        (
            "Stage 2 (Sequential)",
            repo_root / "models" / f"{language}_experimental_stage2_nllb_lora_bf16",
        ),
    ]

    frames = [gather_logs(label, path) for label, path in stage_specs]
    return pd.concat(frames, ignore_index=True)


def plot_losses(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Render the loss curves and persist them to disk."""
    fig, ax = plt.subplots(figsize=(12, 6))

    stage_order = ["Baseline", "Stage 2 (Sequential)"]
    metric_styles = {"train_loss": "-", "eval_loss": "--"}
    colors = {"Baseline": "C0", "Stage 2 (Sequential)": "C2"}

    for stage in stage_order:
        stage_df = df[df["stage"] == stage]
        for metric, style in metric_styles.items():
            metric_df = stage_df[stage_df["metric"] == metric].sort_values("step")
            if metric_df.empty:
                continue
            label = f"{stage} - {metric.replace('_', ' ').title()}"
            ax.plot(
                metric_df["step"],
                metric_df["value"],
                linestyle=style,
                marker="o",
                markersize=4,
                label=label,
                color=colors[stage],
            )

    # Get unique step/epoch pairs for x-axis labels
    unique_points = df.drop_duplicates(subset=["step"])[["step", "epoch"]]
    unique_points = unique_points.sort_values("step")
    
    # Set x-axis ticks and labels
    ax.set_xticks(unique_points["step"].tolist())
    ax.set_xticklabels(
        [f"{int(step)}\n(E{epoch:.0f})" for step, epoch in unique_points.to_numpy()],
        rotation=0,
        ha="center",
    )

    ax.set_xlabel("Training Step (Epoch)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"

    languages = ["waray", "cebuano"]

    for language in languages:
        print(f"Processing {language.capitalize()}...")
        try:
            dataset = build_dataset(repo_root, language)
            plot_path = results_dir / f"{language}_training_eval_loss.png"
            title = f"{language.capitalize()} NLLB LoRA Training and Evaluation Loss"
            plot_losses(dataset, plot_path, title)
            print(f"  ✓ Saved loss curves to {plot_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    main()
