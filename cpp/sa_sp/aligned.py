"""Reuse the received SP decoder in SA with the shared board and objective."""

import math
import random
import time

from cpp.sa_sp.adapter import export_positions, load_original
from integration.evaluation.protocol import normalized_cost, score_layout


def shelf_sequences(circuit, width):
    names, rows, used = list(circuit["blocks"]), [], []
    order = sorted(range(len(names)), key=lambda i: -circuit["blocks"][names[i]][1])
    for index in order:
        w = circuit["blocks"][names[index]][0]
        row = next((i for i, filled in enumerate(used) if filled + w <= width), len(rows))
        if row == len(rows):
            rows.append([])
            used.append(0)
        rows[row].append(index)
        used[row] += w
    return [sum(rows, []), sum(reversed(rows), [])]


def propose(pair, rng):
    first, second = pair
    if len(first) < 2:
        return pair, {"sequence": 0, "indices": []}
    selected = rng.randrange(2)
    changed = list(pair[selected])
    i, j = rng.sample(range(len(first)), 2)
    changed[i], changed[j] = changed[j], changed[i]
    return ([changed, second] if selected == 0 else [first, changed]), {"sequence": selected, "indices": [i, j]}


def decode(module, circuit, protocol, pair):
    names = list(circuit["blocks"])
    x, y, widths, heights = module.sp_to_placement(*pair, circuit["blocks"], names)
    width, height = module.compute_bounding_box(x, y, widths, heights)
    area = float((max(x + widths) - min(x)) * (max(y + heights) - min(y)))
    hpwl = float(module.compute_hpwl(x, y, circuit["nets"], {name: i for i, name in enumerate(names)}))
    return {"pair": pair, "arrays": (x, y, widths, heights), "bounds": [width, height],
            "area": area, "hpwl": hpwl, "cost": normalized_cost(area, hpwl, circuit, protocol),
            "feasible": width <= protocol["width"] and height <= protocol["height"]}


def snapshot(state, circuit):
    return {"sequence_pair": state["pair"], "placements": export_positions(
        circuit, list(circuit["blocks"]), *state["arrays"]),
        **{key: state[key] for key in ("bounds", "area", "hpwl", "cost", "feasible")}}


def initial_temperature(module, circuit, protocol, state, rng):
    values, rejected = [state["cost"]], 0
    for _ in range(max(50, 5*len(circuit["blocks"]))):
        pair, _ = propose(state["pair"], rng)
        candidate = decode(module, circuit, protocol, pair)
        if candidate["feasible"]:
            state = candidate
            values.append(state["cost"])
        else:
            rejected += 1
    maximum = max((abs(b-a) for a, b in zip(values, values[1:])), default=0.0)
    return max(maximum / math.log(2), 1e-8), {"max_normalized_delta": maximum,
                                            "probe_count": max(50, 5*len(circuit["blocks"])),
                                            "infeasible_probes": rejected}


def accept_proposal(delta, temperature, feasible, rng):
    if not feasible:
        return False, 0.0, None
    if delta < 0:
        return True, 1.0, None
    probability = math.exp(-delta / temperature)
    draw = rng.random()
    return draw < probability, probability, draw


def record_event(context, before, candidate, move, delta, accepted, probability, draw, temperature):
    context["trace"].append({"iteration": context["moves"], "temperature": temperature,
                             "move": move, "delta": delta, "accepted": accepted,
                             "acceptance_probability": probability, "random_draw": draw,
                             "before": snapshot(before, context["circuit"]),
                             "candidate": snapshot(candidate, context["circuit"]),
                             "after": snapshot(context["current"], context["circuit"]),
                             "best_cost": context["best"]["cost"]})


def temperature_step(context, temperature, moves):
    for _ in range(moves):
        before = context["current"]
        pair, move = propose(before["pair"], context["rng"])
        candidate = decode(context["module"], context["circuit"], context["protocol"], pair)
        delta = candidate["cost"] - before["cost"]
        accepted, probability, draw = accept_proposal(delta, temperature, candidate["feasible"], context["rng"])
        context["moves"] += 1
        context["infeasible"] += not candidate["feasible"]
        context["accepted"] += accepted
        if accepted:
            context["current"] = candidate
        improved = accepted and candidate["cost"] < context["best"]["cost"]
        if improved:
            context["best"] = candidate
        if context["moves"] <= 24 or improved or context["moves"] == context["budget"]:
            record_event(context, before, candidate, move, delta, accepted, probability, draw, temperature)


def prepare_context(circuit, protocol, seed, stages, moves_per_temperature):
    module, rng = load_original(), random.Random(seed)
    state = decode(module, circuit, protocol, shelf_sequences(circuit, protocol["width"]))
    if not state["feasible"]:
        raise ValueError("The documented shelf initialization does not fit this canvas")
    temperature, probe = initial_temperature(module, circuit, protocol, state, rng)
    moves = moves_per_temperature or max(20*len(circuit["blocks"]), 200)
    return {"module": module, "rng": rng, "circuit": circuit, "protocol": protocol,
            "current": state, "best": state, "initial": snapshot(state, circuit),
            "temperature": temperature, "probe": probe, "moves_per_temperature": moves,
            "moves": 0, "accepted": 0, "infeasible": 0, "trace": [], "budget": stages*moves}


def run_aligned(circuit, protocol, seed=0, stages=180, moves_per_temperature=None, on_stage=None):
    started, cpu = time.perf_counter(), time.process_time()
    context = prepare_context(circuit, protocol, seed, stages, moves_per_temperature)
    history = []
    for stage in range(stages):
        temperature = context["temperature"] * 0.95**stage
        temperature_step(context, temperature, context["moves_per_temperature"])
        history.append({"stage": stage+1, "temperature": temperature, "moves": context["moves"],
                        "current_cost": context["current"]["cost"], "best_cost": context["best"]["cost"],
                        "accepted": context["accepted"], "infeasible": context["infeasible"]})
        if on_stage:
            on_stage(history[-1])
    result = export_result(context, history, seed, stages)
    result["timing"] = {"wall_seconds": time.perf_counter()-started,
                        "process_cpu_seconds": time.process_time()-cpu,
                        "includes": "initialization, normalized-cost probes, search, trace capture and final validation"}
    return result


def export_result(context, history, seed, stages):
    final = snapshot(context["best"], context["circuit"])
    validation = score_layout(context["circuit"], final["placements"], context["protocol"])
    if not validation["legal"] or abs(validation["normalized_cost"]-final["cost"]) > 1e-9:
        raise AssertionError("Independent scoring disagrees with the aligned SA solver")
    return {"algorithm": "sa_sp_aligned", "seed": seed, "initial": context["initial"],
            "final": final, "placements": final["placements"], "validation": validation,
            "moves": context["moves"], "accepted": context["accepted"], "infeasible": context["infeasible"],
            "settings": {"stages": stages, "cooling_factor": 0.95,
                         "moves_per_temperature": context["moves_per_temperature"],
                         "initial_temperature": context["temperature"], **context["probe"]},
            "history": history, "trace": context["trace"],
            "source": "cpp/sa_sp/seqpair_usc.py: sp_to_placement and compute_hpwl"}
