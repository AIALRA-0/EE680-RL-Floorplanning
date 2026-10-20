"""Expose one real search check at a time, then execute and explain the action."""

from copy import deepcopy

from src.environment.placement_env import PlacementEnv


class LearningSession:
    def __init__(self, circuit, width=64, height=64):
        self.env = PlacementEnv(circuit, width, height)
        self.modules = []
        self.events = []
        self.transitions = []
        self.phase = "observe"
        self.queue = []
        self.current = None
        self.commands = []
        self._begin_module()

    def _event(self, kind, **values):
        event = {"number": len(self.events), "kind": kind,
                 "module": self.module["name"], **deepcopy(values)}
        self.events.append(event)
        self.current = event

    @property
    def module(self):
        return self.modules[-1]

    def _begin_module(self):
        self.modules.append({"name": self.env.current_macro,
                             "before": self.env.observation(), "attempts": [], "chosen": None,
                             "mask": [None] * (self.env.width * self.env.height)})
        self.phase = "done" if self.env.terminated else "observe"
        self._event(self.phase, observation=self.env.observation())

    def _candidate(self):
        index = len(self.module["attempts"])
        if index >= self.env.width * self.env.height:
            raise RuntimeError("Search exhausted without the environment reporting failure")
        x, y = index % self.env.width, index // self.env.width
        report = self.env.inspect_action(x, y)
        attempt = {"index": index, "x": x, "y": y, "report": report,
                   "shown_checks": 0, "resolved": False}
        self.module["attempts"].append(attempt)
        self.queue = [deepcopy(check) for check in report["checks"]] + [{"kind": "mask", "legal": report["legal"]}]
        self.phase = "candidate"
        self._event("candidate", index=index, x=x, y=y)

    def advance(self):
        if self.phase == "done":
            return
        attempt = self.module["attempts"][-1] if self.module["attempts"] else None
        if self.phase in ("observe", "reject"):
            self._candidate()
        elif self.queue:
            check = self.queue.pop(0)
            self.phase = check["kind"]
            if self.phase != "mask":
                attempt["shown_checks"] += 1
            else:
                self.module["mask"][attempt["index"]] = attempt["report"]["legal"]
                attempt["resolved"] = True
            self._event(self.phase, index=attempt["index"], x=attempt["x"], y=attempt["y"], check=check)
        elif self.phase == "mask":
            self.phase = "choose" if self.module["mask"][attempt["index"]] else "reject"
            if self.phase == "choose":
                self.module["chosen"] = [attempt["x"], attempt["y"]]
            self._event(self.phase, index=attempt["index"], x=attempt["x"], y=attempt["y"],
                        error=attempt["report"]["error"])
        elif self.phase == "choose":
            before = self.env.observation()
            before_info = self.env.info()
            before_details = self.env.metric_details()
            action = self.module["chosen"]
            observation, reward, terminated, truncated, info = self.env.step(action)
            transition = {"observation": before, "before_info": before_info,
                          "before_details": before_details,
                          "action": action, "reward": reward,
                          "next_observation": observation, "terminated": terminated,
                          "truncated": truncated, "info": info}
            self.transitions.append(deepcopy(transition))
            self.phase = "apply"
            self._event("apply", action=action, observation=observation)
        elif self.phase == "apply":
            self.phase = "metrics"
            self._event("metrics", details=self.env.metric_details())
        elif self.phase == "metrics":
            self.phase = "reward"
            self._event("reward", reward=self.env.reward, terminated=self.env.terminated,
                        reason=self.env.reason, area_reference=self.env.width * self.env.height,
                        hpwl_reference=max(1, len(self.env.circuit["nets"]) * (self.env.width + self.env.height)))
        elif self.phase == "reward":
            self.phase = "next_observation"
            self._event("next_observation", observation=self.env.observation())
        elif self.phase == "next_observation":
            if self.env.terminated:
                self.phase = "done"
                self._event("done", observation=self.env.observation(), reason=self.env.reason)
            else:
                self._begin_module()

    def command(self, name):
        if name not in ("advance", "candidate", "choose", "module"):
            raise ValueError("Unknown learning command")
        module = len(self.modules)
        if name == "advance":
            self.advance()
        else:
            while self.phase != "done":
                self.advance()
                if name == "candidate" and self.phase in ("reject", "choose", "observe"):
                    break
                if name == "choose" and self.phase in ("choose", "next_observation", "observe"):
                    break
                if name == "module" and len(self.modules) != module:
                    break
        self.commands.append(name)
        return self.snapshot()

    def attempt(self, module_index, index):
        module = self.modules[module_index]
        attempt = deepcopy(module["attempts"][index])
        attempt["observation"] = deepcopy(module["before"])
        return attempt

    def snapshot(self):
        attempts = self.module["attempts"]
        return {
            "phase": self.phase, "observation": self.env.observation(),
            "info": self.env.info(), "reward": self.env.reward,
            "details": self.env.metric_details(), "current": deepcopy(self.current),
            "module_index": len(self.modules) - 1, "active_macro": self.module["name"],
            "attempt": deepcopy(attempts[-1]) if attempts else None,
            "attempts": [{"index": a["index"], "x": a["x"], "y": a["y"],
                          "legal": a["report"]["legal"] if a["resolved"] else None,
                          "error": a["report"]["error"] if a["resolved"] else None} for a in attempts],
            "chosen": self.module["chosen"], "event_count": len(self.events),
            "mask": list(self.module["mask"]),
            "recent_events": deepcopy(self.events[-12:]),
            "transitions": deepcopy(self.transitions),
            "modules": [{"name": m["name"], "attempts": len(m["attempts"]),
                         "chosen": m["chosen"]} for m in self.modules],
        }
