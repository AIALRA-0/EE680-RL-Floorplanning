"""Expose the latest local PPO run without importing PyTorch into the server."""

import json
import gzip


def record_exists(path):
    return path.exists() or path.with_suffix(path.suffix + ".gz").exists()


def read_record(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with gzip.open(path.with_suffix(path.suffix + ".gz"), "rt", encoding="utf-8") as stream:
        return json.load(stream)


def available_runs(root, pointer):
    registry = pointer.parent / "ppo-runs.json"
    entries = json.loads(registry.read_text(encoding="utf-8")).get("runs", []) if registry.exists() else []
    if pointer.exists():
        entries.append({**json.loads(pointer.read_text(encoding="utf-8")), "label": "Original small-circuit training"})
    result = {}
    for item in entries:
        directory = (root / item["directory"]).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("PPO output must be inside the repository")
        progress = directory / "progress.json"
        status = json.loads(progress.read_text(encoding="utf-8"))["status"] if progress.exists() else "preparing"
        result.setdefault(item["directory"], {**item, "status": status})
    return list(result.values())


def selected_directory(root, pointer, runs, run):
    chosen = run or json.loads(pointer.read_text(encoding="utf-8"))["directory"]
    if chosen not in {item["directory"] for item in runs}:
        raise ValueError("Choose a registered PPO run")
    directory = (root / chosen).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("PPO output must be inside the repository")
    return directory


def read_run(root, pointer, replay_index=None, update=None, run=None):
    root = root.resolve()
    if not pointer.exists():
        return {"available": False, "message": "No PPO records are available yet. Run the documented training command first."}
    runs = available_runs(root, pointer)
    directory = selected_directory(root, pointer, runs, run)
    if replay_index is not None:
        if replay_index < 0:
            raise ValueError("Replay index must be nonnegative")
        suffix = "" if update is None else f"-update-{update}"
        return read_record(directory / f"replay-{replay_index}{suffix}.json")
    if not (directory / "progress.json").exists():
        return {"available": False, "runs": runs, "message": "This experiment is preparing its initial evaluation. Refresh shortly."}
    result = {"available": True, "directory": directory.relative_to(root).as_posix(), "runs": runs}
    result["checkpoints"] = sorted({int(p.name.split(".json")[0].rsplit("-", 1)[1])
                                   for p in directory.glob("replay-0-update-*.json*")})
    result["figures"] = [p.relative_to(root).as_posix() for p in (directory / "figures").glob("*.png")]
    for name in ("progress", "initial-evaluation", "rollout", "study"):
        path = directory / f"{name}.json"
        result[name] = read_record(path) if record_exists(path) else None
    attach_selected_records(result, directory, update)
    return result


def attach_selected_records(result, directory, update):
    result["selected_update"] = result["progress"]["update"] if update is None else update
    rollout_path = directory / f"rollout-update-{update}.json" if update is not None else directory / "rollout.json"
    if record_exists(rollout_path):
        result["rollout"] = read_record(rollout_path)
    elif result["rollout"] and result["rollout"]["update"] != result["selected_update"]:
        result["rollout"] = None
    teaching = directory / f"teaching-update-{result['selected_update']}.json"
    if not record_exists(teaching):
        teaching = directory / "teaching.json"
    result["teaching"] = read_record(teaching) if record_exists(teaching) else None
