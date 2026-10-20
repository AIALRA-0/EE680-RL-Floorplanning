"""Apply the same physical rules and normalized objective to both solvers."""

from integration.evaluation.placement import evaluate_placement


def comparison_protocol(width, height, area_weight=0.5):
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("The shared board must have positive integer dimensions")
    if not 0 <= area_weight <= 1:
        raise ValueError("Area weight must be in [0, 1]")
    return {"width": width, "height": height, "area_weight": area_weight,
            "rotation": False, "coordinates": "integer lower-left anchors",
            "area": "tight occupied bounding box", "hpwl": "lower-left endpoints",
            "touching_edges": True}


def normalized_cost(area, hpwl, circuit, protocol):
    width, height, alpha = (protocol[k] for k in ("width", "height", "area_weight"))
    return (alpha * area / (width * height)
            + (1 - alpha) * hpwl / max(1, len(circuit["nets"]) * (width + height)))


def score_layout(circuit, placements, protocol):
    result = evaluate_placement(circuit, placements, protocol["width"], protocol["height"])
    for name, position in placements.items():
        rotated = position.get("rotated", False)
        if type(rotated) is not bool or rotated:
            result["errors"].append(f"Rotation is disabled: {name}")
        for axis in ("x", "y"):
            value = position.get(axis)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value % 1:
                result["errors"].append(f"Coordinate must be an integer anchor: {name}.{axis}")
    result["legal"] = not result["errors"]
    cost = normalized_cost(result["area"], result["hpwl"], circuit, protocol) if result["legal"] else None
    return {**result, "normalized_cost": cost, "reward": -cost if cost is not None else -2.0}
