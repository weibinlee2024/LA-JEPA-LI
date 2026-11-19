"""Tripartite attention mask.

Query groups and key groups are ordered ``(observation, future, action)``.

* observation attends only to observation
* future attends to observation and future
* action attends to observation and action

Action queries never read future keys, and observation queries never read
future or action keys. That is what lets inference cache observation K/V and
skip the future tokens entirely.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_tripartite_mask(
    n_obs: int,
    n_fut: int,
    n_act: int,
    obs_keep: torch.Tensor | None = None,
    fut_keep: torch.Tensor | None = None,
    act_keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean mask ``[B, Q, K]`` (or ``[Q, K]`` when no keep-masks are given).

    True means the query is allowed to read that key. Dropped cameras and
    padded actions are additional False entries on top of the block pattern.
    """
    n_q = n_obs + n_fut + n_act
    allowed = torch.zeros(n_q, n_q, dtype=torch.bool)
    # obs -> obs
    allowed[:n_obs, :n_obs] = True
    # fut -> obs + fut
    if n_fut:
        allowed[n_obs : n_obs + n_fut, : n_obs + n_fut] = True
    # act -> obs + act
    if n_act:
        q0 = n_obs + n_fut
        allowed[q0:, :n_obs] = True
        allowed[q0:, n_obs + n_fut :] = True
    if obs_keep is None and fut_keep is None and act_keep is None:
        return allowed
    keeps = []
    for count, keep in ((n_obs, obs_keep), (n_fut, fut_keep), (n_act, act_keep)):
        if count == 0:
            continue
        if keep is None:
            raise ValueError("pass every keep-mask or none of them")
        if keep.shape[-1] != count:
            raise ValueError(f"keep mask length {keep.shape[-1]} != {count}")
        keeps.append(keep)
    key_keep = torch.cat(keeps, dim=-1)
    if key_keep.ndim == 1:
        return allowed & key_keep.unsqueeze(0)
    return allowed.to(key_keep.device).unsqueeze(0) & key_keep.unsqueeze(1)


def attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled-dot-product attention.

    ``query/key/value`` are ``[B, S, H, D]``. ``key_keep`` is ``[B, S_k]``
    with True = visible. A sample whose keys are all hidden gets zeros
    instead of a NaN softmax.
    """
    if key.shape[1] == 0:
        return query.new_zeros(query.shape)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    mask = None
    wiped = None
    if key_keep is not None:
        wiped = ~key_keep.any(dim=-1)
        safe = key_keep
        if wiped.any():
            safe = key_keep.clone()
            safe[wiped, 0] = True
        # True = this key is visible to every query.
        mask = safe[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = out.transpose(1, 2).contiguous()
    if wiped is not None and wiped.any():
        out = out.clone()
        out[wiped] = 0
    return out


def attend_obs(q, k, v, obs_keep):
    return attend(q, k, v, obs_keep)


def attend_future(q_fut, k_obs, v_obs, k_fut, v_fut, obs_keep, fut_keep):
    if q_fut.shape[1] == 0:
        return q_fut
    key_keep = _cat_keep(obs_keep, fut_keep)
    return attend(q_fut, torch.cat([k_obs, k_fut], dim=1), torch.cat([v_obs, v_fut], dim=1), key_keep)


def attend_action(q_act, k_obs, v_obs, k_act, v_act, obs_keep, act_keep):
    key_keep = _cat_keep(obs_keep, act_keep)
    return attend(q_act, torch.cat([k_obs, k_act], dim=1), torch.cat([v_obs, v_act], dim=1), key_keep)


def _cat_keep(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor | None:
    if left is None and right is None:
        return None
    if left is None or right is None:
        raise ValueError("obs/future/action keep-masks must all be set or all be omitted")
    return torch.cat([left, right], dim=-1)
