"""Export plots from actual PPO histories and saved evaluated layouts."""

import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def ablation_curves(directory, output):
    summary = json.loads((directory / "summary.json").read_text())
    weights = sorted({r["area_weight"] for r in summary["runs"]})
    fig, axes = plt.subplots(2, len(weights), figsize=(13, 7), squeeze=False, layout="constrained")
    for column, weight in enumerate(weights):
        runs = [r for r in summary["runs"] if r["area_weight"] == weight and r["returncode"] == 0]
        histories = [json.loads((Path(r["directory"]) / "progress.json").read_text())["history"] for r in runs]
        for row, (key, label) in enumerate((("mean_reward", "Mean training return"),
                                           ("legal_completion_rate", "Legal completion rate"))):
            count = min(map(len, histories))
            values = [[history[i][key] for history in histories] for i in range(count)]
            means = [statistics.mean(v) for v in values]
            spread = [statistics.stdev(v) if len(v) > 1 else 0 for v in values]
            axis = axes[row, column]
            axis.plot(range(1, count + 1), means, color="#303030", linewidth=1.4)
            axis.fill_between(range(1, count + 1), [m-s for m,s in zip(means,spread)],
                              [m+s for m,s in zip(means,spread)], color="#777777", alpha=.22)
            axis.set(xlabel="PPO update", ylabel=label, title=f"Area weight {weight:g}; seeds n={len(runs)}")
            axis.grid(alpha=.18)
            if row == 1:
                axis.set_ylim(0, 1.04)
    fig.suptitle("5/10-macro training: mean and sample standard deviation across seeds\n"
                 "16 episodes/update; 100 updates; reward differs by weight; failures retained")
    fig.savefig(output / "ablation-curves.png", dpi=180)
    plt.close(fig)


def training_curve(directory, output):
    progress = json.loads((directory / "progress.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, key, label in zip(axes, ("mean_reward", "legal_completion_rate"),
                                ("Mean training return", "Legal completion rate")):
        axis.plot([r["update"] for r in progress["history"]],
                  [r[key] for r in progress["history"]], color="#303030", linewidth=1)
        axis.set(xlabel="PPO update", ylabel=label)
        axis.grid(alpha=.18)
    axes[1].set_ylim(0, 1.04)
    names = '/'.join(Path(p).stem for p in progress['inputs'])
    fig.suptitle(f"{names}; seed {progress['seed']}; n=1 training run; {progress.get('episodes_per_update',16)} episodes/update\n"
                 "Actual update statistics; no uncertainty estimate from one seed")
    fig.savefig(output / "training-curve.png", dpi=180)
    plt.close(fig)


def floorplans(directory, output):
    progress = json.loads((directory / "progress.json").read_text())
    replays = [json.loads((directory / f"replay-{i}.json").read_text()) for i in range(len(progress['evaluations']))]
    fig, axes = plt.subplots(1, len(replays), figsize=(6*len(replays), 5), layout="constrained", squeeze=False)
    for axis, replay in zip(axes[0], replays):
        draw_layout(axis, replay)
        m = replay["metrics"]
        axis.set_title(f"{m['total']} macros; legal={replay['validation']['legal']}\n"
                       f"Area {m['area']:g}; HPWL {m['hpwl']:g}; trace inference {replay['inference_seconds']:.3f} s")
    fig.suptitle(f"Saved PPO argmax layouts; {progress['width']} x {progress['height']} canvas; no rotation\n"
                 "All pairwise net connections use lower-left corners; same input coordinate scale")
    fig.savefig(output / "ppo-floorplans.png", dpi=180)
    plt.close(fig)


def draw_layout(axis, replay):
    positions, blocks = replay["placements"], replay["circuit"]["blocks"]
    for net in replay["circuit"]["nets"]:
        for i, name in enumerate(net):
            for other in net[i+1:]:
                if name in positions and other in positions:
                    a, b = positions[name], positions[other]
                    axis.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#777777", alpha=.45, linewidth=.6)
    for i, (name, p) in enumerate(positions.items()):
        w, h = blocks[name]
        axis.add_patch(Rectangle((p["x"], p["y"]), w, h, facecolor=str(.90 if i%2 else .96),
                                 edgecolor="#333333", linewidth=.8, alpha=.8))
        axis.text(p["x"]+w/2, p["y"]+h/2, name, ha="center", va="center", fontsize=8)
        axis.plot(p["x"], p["y"], ".", color="#333333", markersize=2)
    axis.set(xlim=(0, replay["board"]["width"]), ylim=(0, replay["board"]["height"]),
             xlabel="x (input units)", ylabel="y (input units)")
    axis.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training_curve(args.run, args.output)
    if args.ablation:
        ablation_curves(args.ablation, args.output)
    floorplans(args.run, args.output)
    print("Exported actual training curves and labeled floorplans" + (" with ablation curves" if args.ablation else ""))


if __name__ == "__main__":
    main()
