"""Collect on-policy episodes and apply PPO's clipped objective with GAE."""

from dataclasses import asdict, dataclass
import torch

from src.environment.placement_env import PlacementEnv
from src.ppo.encoding import collate, encode_observation


@dataclass
class PPOConfig:
    learning_rate: float = 0.001
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch: int = 32
    target_kl: float = 0.03


def generalized_advantages(rewards, values, next_values, terminated, boundaries, gamma=1.0, lam=0.95):
    """Bootstrap time limits, but stop the recursion at every episode boundary."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (~terminated[i]) - values[i]
        carry = delta + gamma * lam * (~boundaries[i]) * carry
        advantages[i] = carry
    return advantages, advantages + values


@torch.no_grad()
def collect_episodes(model, circuits, episodes, width=64, height=64, area_weight=0.5, start=0):
    records, summaries = [], []
    for episode in range(episodes):
        env = PlacementEnv(circuits[(start + episode) % len(circuits)], width, height, area_weight)
        if env.terminated:
            raise ValueError("Circuit has no legal initial placement on this grid")
        while not env.terminated:
            observation = env.observation()
            encoded = encode_observation(observation, env.legal_mask())
            distribution, value = model.distribution(collate([encoded]))
            action = distribution.sample()
            next_observation, reward, done, truncated, info = env.step((int(action) % width, int(action) // width))
            records.append({"encoded": encoded, "action": int(action),
                            "log_prob": float(distribution.log_prob(action)), "value": float(value),
                            "reward": reward, "terminated": done, "boundary": done or truncated,
                            "macro": observation["current_macro"],
                            "circuit_index": (start + episode) % len(circuits), "episode": episode,
                            "placements_before": observation["placements"],
                            "placements_after": next_observation["placements"], "metrics_after": info["metrics"],
                            "next_macro": next_observation["current_macro"]})
        summaries.append({**env.metrics(), "reward": env.reward, "reason": env.reason})
    return records, summaries


def rollout_tensors(records, config, include_details=False):
    values = torch.tensor([r["value"] for r in records])
    terminated = torch.tensor([r["terminated"] for r in records])
    boundaries = torch.tensor([r["boundary"] for r in records])
    next_values = torch.cat((values[1:], torch.zeros(1)))
    advantages, returns = generalized_advantages(torch.tensor([r["reward"] for r in records]),
                                                values, next_values, terminated, boundaries,
                                                config.gamma, config.gae_lambda)
    raw = advantages
    mean, std = raw.mean(), raw.std(unbiased=False).clamp_min(1e-8)
    advantages = (raw - mean) / std
    result = {"actions": torch.tensor([r["action"] for r in records]),
            "old_log_prob": torch.tensor([r["log_prob"] for r in records]),
            "advantages": advantages, "returns": returns}
    if include_details:
        result.update({"advantages_raw": raw, "advantage_mean": mean, "advantage_std": std})
    return result


def ppo_loss(distribution, values, actions, old_log_prob, advantages, returns, config):
    log_ratio = distribution.log_prob(actions) - old_log_prob
    ratio = log_ratio.exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1 - config.clip, 1 + config.clip) * advantages).mean()
    value = 0.5 * (values - returns).square().mean()
    entropy = distribution.entropy().mean()
    loss = policy + config.value_coefficient * value - config.entropy_coefficient * entropy
    stats = {"policy_loss": policy, "value_loss": value, "entropy": entropy,
             "approx_kl": ((ratio - 1) - log_ratio).mean(),
             "clip_fraction": ((ratio - 1).abs() > config.clip).float().mean()}
    return loss, stats


def update_example(distribution, value, selected, record, config):
    old, new = float(selected["old_log_prob"][0]), float(distribution.log_prob(selected["actions"])[0].detach())
    ratio, advantage = float(torch.tensor(new - old).exp()), float(selected["advantages"][0])
    clipped = min(1 + config.clip, max(1 - config.clip, ratio))
    return {"episode": record["episode"], "macro": record["macro"], "action": record["action"],
            "circuit_index": record["circuit_index"],
            "old_log_prob": old, "log_prob_before_step": new, "ratio": ratio,
            "advantage_normalized": advantage, "return": float(selected["returns"][0]),
            "value_before_step": float(value[0].detach()),
            "entropy": float(distribution.entropy()[0].detach()),
            "unclipped_objective": ratio * advantage, "clipped_objective": clipped * advantage,
            "selected_objective": min(ratio * advantage, clipped * advantage), "clip": config.clip}


def minibatch_step(model, optimizer, records, indices, tensors, config, capture=False):
    batch = collate([records[int(i)]["encoded"] for i in indices])
    distribution, value = model.distribution(batch)
    selected = {key: tensor[indices] for key, tensor in tensors.items()}
    loss, stats = ppo_loss(distribution, value, config=config, **selected)
    if not torch.isfinite(loss):
        raise FloatingPointError("PPO loss is not finite")
    example = update_example(distribution, value, selected, records[int(indices[0])], config) if capture else None
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
    if capture:
        example["gradient_norm_before_clip"] = float(norm)
        example["gradient_norm_after_clip"] = float(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm())
        example["minibatch_stats"] = {**{key: float(v.detach()) for key, v in stats.items()}, "loss": float(loss.detach())}
    optimizer.step()
    if capture:
        with torch.no_grad():
            after, after_value = model.distribution(collate([records[int(indices[0])]["encoded"]]))
            example.update({"log_prob_after_step": float(after.log_prob(selected["actions"][:1])),
                            "value_after_step": float(after_value)})
    return {**{key: float(v.detach()) for key, v in stats.items()},
            "loss": float(loss.detach()), "gradient_norm": float(norm)}, example


def update_policy(model, optimizer, records, config):
    tensors = rollout_tensors(records, config)
    before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    history = []
    example = None
    for _ in range(config.epochs):
        for indices in torch.randperm(len(records)).split(config.minibatch):
            row, current_example = minibatch_step(model, optimizer, records, indices, tensors, config, example is None)
            history.append(row)
            example = example or current_example
            if history[-1]["approx_kl"] > config.target_kl:
                break
        if history[-1]["approx_kl"] > config.target_kl:
            break
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    return {**{key: sum(row[key] for row in history) / len(history) for key in history[0]},
            "parameter_delta_l2": float((after - before).norm()), "optimizer_steps": len(history),
            "example": example}


def save_checkpoint(path, model, optimizer, config, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.pt")
    torch.save({"format": 1, "model": model.state_dict(), "hidden": model.hidden,
                "optimizer": optimizer.state_dict(), "config": asdict(config),
                "metadata": metadata, "torch_rng": torch.get_rng_state()}, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    from src.ppo.model import ActorCritic
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != 1:
        raise ValueError("Unsupported checkpoint format")
    model = ActorCritic(saved["hidden"])
    model.load_state_dict(saved["model"])
    config = PPOConfig(**saved["config"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    optimizer.load_state_dict(saved["optimizer"])
    return model, optimizer, config, saved
