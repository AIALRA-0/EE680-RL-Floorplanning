"""Validate actual archived bytes, teaching arithmetic and comparison exports."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from integration.evaluation.protocol import score_layout
from integration.visualization.ppo_data import read_record


def check_transition(row, following, data):
    config = data["config"]
    delta = row["reward"] + config["gamma"] * row["next_value"] * (not row["terminated"]) - row["value"]
    assert math.isclose(row["delta"], delta, abs_tol=1e-6)
    carry = following["advantage_raw"] if not row["boundary"] else 0
    advantage = delta + config["gamma"] * config["gae_lambda"] * (not row["boundary"]) * carry
    assert math.isclose(row["advantage_raw"], advantage, abs_tol=1e-5)
    assert math.isclose(row["return"], row["advantage_raw"] + row["value"], abs_tol=1e-5)
    assert math.isclose(row["advantage_normalized"], (row["advantage_raw"] - data["advantage_mean"]) / data["advantage_std"], abs_tol=2e-4)
    before, after = row["placements_before"], row["placements_after"]
    assert len(after) == len(before) + 1 and row["macro"] not in before
    assert all(after[k] == value for k, value in before.items())
    if not row["boundary"]:
        assert after == following["placements_before"]
    assert row["terminated"] == row["boundary"]
    if row["terminated"]:
        assert row["next_value"] == 0
    circuit = data["circuits"][row["circuit_index"]]
    w, h = circuit["blocks"][row["macro"]]
    pos = after[row["macro"]]
    assert pos["x"] == row["action"] % data["board"]["width"]
    assert pos["y"] == row["action"] // data["board"]["width"]
    assert 0 <= pos["x"] <= data["board"]["width"] - w
    assert 0 <= pos["y"] <= data["board"]["height"] - h
    for name, p in before.items():
        ow, oh = circuit["blocks"][name]
        assert not (pos["x"] < p["x"] + ow and p["x"] < pos["x"] + w and pos["y"] < p["y"] + oh and p["y"] < pos["y"] + h)


def check_teaching(directory):
    updates, transitions = 0, 0
    for file in directory.glob("teaching-update-*.json*"):
        plain = file.with_suffix("") if file.suffix == ".gz" else file
        data = read_record(plain)
        update = int(plain.stem.rsplit("-", 1)[1])
        assert data["source_update"] == update
        assert (directory / f"checkpoint-{update}.pt").exists() and plain.with_suffix(".pt").exists()
        rows = data["transitions"]
        for i, row in enumerate(rows):
            check_transition(row, rows[i+1] if i+1 < len(rows) else None, data)
        example = data["update_stats"]["example"]
        assert math.isclose(example["ratio"], math.exp(example["log_prob_before_step"] - example["old_log_prob"]), rel_tol=1e-6)
        assert math.isclose(example["selected_objective"], min(example["unclipped_objective"], example["clipped_objective"]), abs_tol=1e-7)
        assert example["gradient_norm_after_clip"] <= data["config"]["max_grad_norm"] + 1e-5
        assert data["update_stats"]["parameter_delta_l2"] > 0
        updates += 1
        transitions += len(rows)
    return updates, transitions


def check_comparison():
    path = ROOT / "steps/evidence/comparison-20261020/summary.json"
    assert path.exists(), "The published comparison summary is required"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert [case["macros"] for case in summary["cases"]] == [5, 10, 20, 50, 100]
    trials = 0
    for case in summary["cases"]:
        for entry in case["trials"]:
            trial = json.loads((ROOT / entry["file"]).read_text(encoding="utf-8"))
            assert score_layout(trial["circuit"], trial["placements"], trial["protocol"]) == trial["validation"], entry["file"]
            if trial["algorithm"] == "sa_sp_aligned":
                costs = [row["best_cost"] for row in trial["history"]]
                assert costs == sorted(costs, reverse=True)
                assert trial["trace"][-1]["iteration"] == trial["moves"]
                for event in trial["trace"]:
                    assert event["after"] == (event["candidate"] if event["accepted"] else event["before"])
            trials += 1
    assert trials == 35
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("steps/build/evidence-validation.json"))
    args = parser.parse_args()
    manifest = json.loads((ROOT / "steps/evidence/manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        raw = (ROOT / entry["file"]).read_bytes()
        if entry["file"].endswith(".json.gz"):
            raw = gzip.decompress(raw)
        assert len(raw) == entry["source_bytes"] and hashlib.sha256(raw).hexdigest() == entry["source_sha256"], entry["file"]
    registry = json.loads((ROOT / "steps/evidence/ppo-runs.json").read_text(encoding="utf-8"))
    totals = [check_teaching(ROOT / item["directory"]) for item in registry["runs"]]
    assert sum(row[0] for row in totals) == 18 and sum(row[1] for row in totals) == 2077
    result = {"result": "PASS", "manifest_files": len(manifest["files"]), "runs": len(totals),
              "teaching_updates": sum(row[0] for row in totals), "transitions": sum(row[1] for row in totals),
              "comparison_trials": check_comparison()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
