"""Export deterministic placement data; the maintained UI is steps/index.html."""


def placement_trace(env):
    """Record first-legal actions and Python-scored candidates at each state."""
    states = []
    while True:
        mask = env.legal_mask()
        candidates = []
        for index, legal in enumerate(mask):
            if legal:
                metrics = env.preview(index % env.width, index // env.width)["metrics"]
                candidates.append([metrics["area"], metrics["hpwl"]])
            else:
                candidates.append(None)
        action_index = next((i for i, legal in enumerate(mask) if legal), None)
        action = None if action_index is None else [action_index % env.width, action_index // env.width]
        states.append({"observation": env.observation(), "info": env.info(),
                       "reward": env.reward, "candidates": candidates, "next_action": action})
        if env.terminated:
            break
        env.step(action)
    return {"policy": "first_legal_anchor", "states": states}
