"""Encode geometry, hypernets and candidate costs without using future positions."""

import torch

NODE_FEATURES = ("width/W", "height/H", "area/WH", "x/W", "y/H",
                 "placed", "current", "incident_nets/M")
CANDIDATE_FEATURES = ("x/W", "y/H", "next_area/WH", "delta_hpwl/reference",
                      "next_bbox_width/W", "next_bbox_height/H")


def candidate_features(observation):
    board, circuit, positions = (observation[k] for k in ("board", "circuit", "placements"))
    width, height = board["width"], board["height"]
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    xs, ys = xs.flatten().float(), ys.flatten().float()
    macro = observation["current_macro"]
    mw, mh = circuit["blocks"][macro]
    left, bottom, right, top = xs, ys, xs + mw, ys + mh
    if positions:
        left = xs.minimum(torch.tensor(min(p["x"] for p in positions.values())))
        bottom = ys.minimum(torch.tensor(min(p["y"] for p in positions.values())))
        right = right.maximum(torch.tensor(max(p["x"] + circuit["blocks"][n][0] for n, p in positions.items())))
        top = top.maximum(torch.tensor(max(p["y"] + circuit["blocks"][n][1] for n, p in positions.items())))
    bw, bh = right - left, top - bottom
    delta = incremental_hpwl(circuit, positions, macro, xs, ys)
    reference = max(1, len(circuit["nets"]) * (width + height))
    return torch.stack((xs / width, ys / height, bw * bh / (width * height),
                        delta / reference, bw / width, bh / height), dim=-1)


def incremental_hpwl(circuit, positions, macro, xs, ys):
    delta = torch.zeros_like(xs)
    for net in circuit["nets"]:
        placed = [positions[name] for name in net if name in positions]
        if macro not in net or not placed:
            continue
        for coordinates, axis in ((xs, "x"), (ys, "y")):
            lo, hi = min(p[axis] for p in placed), max(p[axis] for p in placed)
            delta += (lo - coordinates).clamp_min(0) + (coordinates - hi).clamp_min(0)
    return delta


def encode_observation(observation, mask):
    if observation["current_macro"] is None or not any(mask):
        raise ValueError("An active state with a legal action is required")
    circuit, board, positions = (observation[k] for k in ("circuit", "board", "placements"))
    names = list(circuit["blocks"])
    width, height = board["width"], board["height"]
    incidence = torch.tensor([[float(name in net) for name in names]
                             for net in circuit["nets"]]).reshape(len(circuit["nets"]), len(names))
    nodes = []
    for i, name in enumerate(names):
        w, h = circuit["blocks"][name]
        position = positions.get(name, {"x": 0, "y": 0})
        nodes.append([w / width, h / height, w * h / (width * height),
                      position["x"] / width, position["y"] / height, float(name in positions),
                      float(name == observation["current_macro"]),
                      float(incidence[:, i].sum()) / max(1, len(circuit["nets"]))])
    return {"nodes": torch.tensor(nodes), "incidence": incidence,
            "candidates": candidate_features(observation), "mask": torch.tensor(mask, dtype=torch.bool),
            "node_mask": torch.ones(len(names)),
            "global": torch.tensor([len(positions) / len(names),
                                     sum(w * h for w, h in circuit["blocks"].values()) / (width * height),
                                     width / (width + height), observation.get("area_weight", 0.5)])}


def collate(states, device="cpu"):
    """Pad only graph dimensions; every state in a batch uses the same grid."""
    if not states:
        raise ValueError("Cannot batch an empty list")
    nodes = torch.nn.utils.rnn.pad_sequence([s["nodes"] for s in states], batch_first=True)
    node_mask = torch.nn.utils.rnn.pad_sequence([s["node_mask"] for s in states], batch_first=True)
    max_n, max_m = nodes.shape[1], max(s["incidence"].shape[0] for s in states)
    incidence = torch.zeros(len(states), max_m, max_n)
    for i, state in enumerate(states):
        m, n = state["incidence"].shape
        incidence[i, :m, :n] = state["incidence"]
    result = {"nodes": nodes, "node_mask": node_mask, "incidence": incidence}
    for key in ("candidates", "mask", "global"):
        result[key] = torch.stack([state[key] for state in states])
    return {key: value.to(device) for key, value in result.items()}
