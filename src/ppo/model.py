"""A small shared graph encoder with masked actor and scalar critic heads."""

import math
import torch
from torch import nn
from torch.distributions import Categorical


def masked_distribution(logits, mask):
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Logits and boolean mask must have matching shapes")
    if not mask.any(dim=-1).all():
        raise ValueError("No legal action; the environment must terminate this episode")
    if not torch.isfinite(logits).all():
        raise ValueError("Policy logits must be finite before masking")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


class ActorCritic(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.node = nn.Sequential(nn.Linear(8, hidden), nn.Tanh())
        self.message = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(2 * hidden + 4, hidden), nn.Tanh())
        self.query = nn.Linear(hidden, hidden)
        self.candidate = nn.Sequential(nn.Linear(6, hidden), nn.Tanh())
        self.actor_bias = nn.Linear(6, 1, bias=False)
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.query.weight, gain=0.01)
        nn.init.zeros_(self.actor_bias.weight)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1)

    def encode_context(self, batch):
        embeddings = self.node(batch["nodes"])
        incidence = batch["incidence"]
        net_mean = incidence @ embeddings / incidence.sum(-1, keepdim=True).clamp_min(1)
        incoming = incidence.transpose(1, 2) @ net_mean
        incoming = incoming / incidence.sum(1).unsqueeze(-1).clamp_min(1)
        embeddings = self.message(torch.cat((embeddings, incoming), dim=-1))
        valid = batch["node_mask"].unsqueeze(-1)
        pooled = (embeddings * valid).sum(1) / valid.sum(1).clamp_min(1)
        current = (embeddings * batch["nodes"][:, :, 6:7]).sum(1)
        context = self.fusion(torch.cat((pooled, current, batch["global"]), dim=-1))
        return context, {"node_embeddings": embeddings, "net_embeddings": net_mean,
                         "pooled": pooled, "current": current}

    def forward(self, batch):
        context, _ = self.encode_context(batch)
        candidates = self.candidate(batch["candidates"])
        logits = (candidates * self.query(context).unsqueeze(1)).sum(-1) / math.sqrt(self.hidden)
        logits = logits + self.actor_bias(batch["candidates"]).squeeze(-1)
        return logits, self.critic(context).squeeze(-1)

    def distribution(self, batch):
        logits, value = self(batch)
        return masked_distribution(logits, batch["mask"]), value
