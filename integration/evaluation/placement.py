"""Independently validate saved layouts and score lower-left net endpoints."""

import math


def rectangle(circuit, name, position):
    width, height = circuit["blocks"][name]
    if position.get("rotated", False):
        width, height = height, width
    return position["x"], position["y"], position["x"] + width, position["y"] + height


def evaluate_placement(circuit, placements, width, height):
    errors, rectangles = [], {}
    for name, position in placements.items():
        if name not in circuit["blocks"]:
            errors.append(f"Unknown macro: {name}")
            continue
        if not all(isinstance(position.get(k), (int, float)) and not isinstance(position[k], bool)
                   and math.isfinite(position[k]) for k in ("x", "y")):
            errors.append(f"Invalid coordinates: {name}")
            continue
        rect = rectangle(circuit, name, position)
        rectangles[name] = rect
        if rect[0] < 0 or rect[1] < 0 or rect[2] > width or rect[3] > height:
            errors.append(f"Outside board: {name}")
    for i, (name, a) in enumerate(rectangles.items()):
        for other, b in list(rectangles.items())[i + 1:]:
            if max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3]):
                errors.append(f"Overlap: {name}, {other}")
    complete = set(placements) == set(circuit["blocks"])
    if not complete:
        errors.append("Placement is incomplete")
    metrics = score_rectangles(circuit, rectangles)
    return {"legal": not errors, "complete": complete, "errors": errors, **metrics}


def score_rectangles(circuit, rectangles):
    area, hpwl = 0.0, 0.0
    if rectangles:
        area = ((max(r[2] for r in rectangles.values()) - min(r[0] for r in rectangles.values()))
                * (max(r[3] for r in rectangles.values()) - min(r[1] for r in rectangles.values())))
    per_net = []
    for net in circuit["nets"]:
        points = [rectangles[name] for name in net if name in rectangles]
        wire = sum(max(p[axis] for p in points) - min(p[axis] for p in points)
                   for axis in (0, 1)) if points else 0.0
        hpwl += wire
        per_net.append(wire)
    return {"area": area, "hpwl": hpwl, "per_net_hpwl": per_net}
