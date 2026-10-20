"""Check and score sequential grid placement independently of the policy."""

from copy import deepcopy
import math


class PlacementEnv:
    """Place macros in input order on integer lower-left coordinates."""

    def __init__(self, circuit, width=64, height=64, area_weight=0.5):
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError("Board width and height must be positive integers")
        if not math.isfinite(area_weight) or not 0 <= area_weight <= 1:
            raise ValueError("Area weight must be between zero and one")
        self.area_weight = area_weight
        self.circuit = deepcopy(circuit)
        self.names = list(self.circuit["blocks"])
        if not self.names:
            raise ValueError("Circuit must contain at least one block")
        for name, dimensions in self.circuit["blocks"].items():
            if len(dimensions) != 2 or not all(
                math.isfinite(value) and value > 0 for value in dimensions
            ):
                raise ValueError(f"Invalid dimensions for block {name!r}")
        if any(name not in self.names for net in self.circuit["nets"] for name in net):
            raise ValueError("Net references an unknown block")
        self.width, self.height = width, height
        self.reset()

    @property
    def current_macro(self):
        return self.names[len(self.placements)] if len(self.placements) < len(self.names) else None

    def reset(self):
        self.placements = {}
        self.reward = 0.0
        self.terminated = False
        self.reason = None
        self._finish_if_needed()
        return self.observation(), self.info()

    def observation(self):
        return {
            "circuit": deepcopy(self.circuit),
            "board": {"width": self.width, "height": self.height},
            "placements": deepcopy(self.placements),
            "current_macro": self.current_macro,
            "remaining": self.names[len(self.placements):],
            "area_weight": self.area_weight,
        }

    def info(self):
        return {"terminated": self.terminated, "reason": self.reason, "metrics": self.metrics()}

    def action_error(self, x, y):
        return self.inspect_action(x, y)["error"]

    def inspect_action(self, x, y):
        """Return the actual numeric checks, stopping at the first rejection."""
        checks = []

        def finish(error=None):
            return {"legal": error is None, "error": error, "checks": checks}

        if self.terminated:
            return finish("Episode has already ended")
        if type(x) is not int or type(y) is not int:
            return finish("Coordinates must be integers")
        width, height = self.circuit["blocks"][self.current_macro]
        inside = x >= 0 and y >= 0 and x + width <= self.width and y + height <= self.height
        checks.append({"kind": "bounds", "passed": inside,
                       "x": x, "y": y, "right": x + width, "top": y + height,
                       "board_width": self.width, "board_height": self.height})
        if not inside:
            return finish("Macro would extend outside the board")
        for name, position in self.placements.items():
            other_width, other_height = self.circuit["blocks"][name]
            other_x, other_y = position["x"], position["y"]
            comparisons = [[x, other_x + other_width], [other_x, x + width],
                           [y, other_y + other_height], [other_y, y + height]]
            overlap = all(left < right for left, right in comparisons)
            checks.append({"kind": "overlap", "block": name,
                           "passed": not overlap, "comparisons": comparisons})
            if overlap:
                return finish(f"Macro would overlap {name}")
        return finish()

    def metric_details(self):
        """Expose the bounding box and each net's contribution for the reader."""
        positions = self.placements
        bounds = None
        if positions:
            bounds = {
                "left": min(p["x"] for p in positions.values()),
                "bottom": min(p["y"] for p in positions.values()),
                "right": max(p["x"] + self.circuit["blocks"][name][0] for name, p in positions.items()),
                "top": max(p["y"] + self.circuit["blocks"][name][1] for name, p in positions.items()),
            }
        nets = []
        for index, net in enumerate(self.circuit["nets"]):
            placed = {name: positions[name] for name in net if name in positions}
            xs = [p["x"] for p in placed.values()]
            ys = [p["y"] for p in placed.values()]
            dx = max(xs) - min(xs) if xs else 0
            dy = max(ys) - min(ys) if ys else 0
            nets.append({"index": index, "placed": placed,
                         "missing": [name for name in net if name not in positions],
                         "dx": dx, "dy": dy, "hpwl": dx + dy})
        return {"bounds": bounds, "nets": nets, "metrics": self.metrics()}

    def legal_mask(self):
        """Mark integer anchors using the same strict rectangle inequalities."""
        mask = [False] * (self.width * self.height)
        if self.terminated:
            return mask
        width, height = self.circuit["blocks"][self.current_macro]
        xmax = min(self.width - 1, math.floor(self.width - width))
        ymax = min(self.height - 1, math.floor(self.height - height))
        if min(xmax, ymax) < 0:
            return mask
        for y in range(ymax + 1):
            mask[y * self.width:y * self.width + xmax + 1] = [True] * (xmax + 1)
        for name, position in self.placements.items():
            other_width, other_height = self.circuit["blocks"][name]
            x0 = max(0, math.floor(position["x"] - width) + 1)
            x1 = min(xmax, math.ceil(position["x"] + other_width) - 1)
            y0 = max(0, math.floor(position["y"] - height) + 1)
            y1 = min(ymax, math.ceil(position["y"] + other_height) - 1)
            if x0 <= x1:
                for y in range(y0, y1 + 1):
                    mask[y * self.width + x0:y * self.width + x1 + 1] = [False] * (x1 - x0 + 1)
        return mask

    def metrics(self, placements=None):
        positions = self.placements if placements is None else placements
        if positions:
            left = min(p["x"] for p in positions.values())
            bottom = min(p["y"] for p in positions.values())
            right = max(p["x"] + self.circuit["blocks"][name][0] for name, p in positions.items())
            top = max(p["y"] + self.circuit["blocks"][name][1] for name, p in positions.items())
            area = (right - left) * (top - bottom)
        else:
            area = 0.0
        hpwl = 0.0
        for net in self.circuit["nets"]:
            points = [positions[name] for name in net if name in positions]
            if points:
                hpwl += max(p["x"] for p in points) - min(p["x"] for p in points)
                hpwl += max(p["y"] for p in points) - min(p["y"] for p in points)
        return {"area": area, "hpwl": hpwl, "placed": len(positions),
                "total": len(self.names), "complete": len(positions) == len(self.names)}

    def preview(self, x, y):
        """Score a candidate without changing the environment."""
        error = self.action_error(x, y)
        if error:
            return {"legal": False, "error": error, "metrics": None}
        candidate = dict(self.placements)
        candidate[self.current_macro] = {"x": x, "y": y, "rotated": False}
        return {"legal": True, "error": None, "metrics": self.metrics(candidate)}

    def _finish_if_needed(self):
        if self.current_macro is None:
            self.terminated, self.reason = True, "complete"
            metrics = self.metrics()
            area_reference = self.width * self.height
            hpwl_reference = max(1, len(self.circuit["nets"]) * (self.width + self.height))
            self.reward = (-self.area_weight * metrics["area"] / area_reference
                           - (1 - self.area_weight) * metrics["hpwl"] / hpwl_reference)
        elif not any(self.legal_mask()):
            self.terminated, self.reason, self.reward = True, "no_legal_action", -2.0

    def step(self, action):
        """Return observation, reward, terminated, truncated, and information."""
        x, y = action
        error = self.action_error(x, y)
        if error:
            raise ValueError(error)
        self.placements[self.current_macro] = {"x": x, "y": y, "rotated": False}
        self.reward = 0.0
        self._finish_if_needed()
        return self.observation(), self.reward, self.terminated, False, self.info()
