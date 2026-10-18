"""
Cosine-similarity computations between control and experiment contrast directions.

The four activation pickles produced by the cache stage have shape
``(k_per_class, n_positions, n_layers, d_model)`` per
``(condition, GT_label)`` split (``k_per_class`` is the achieved per-class
count for the configured ``cache_sample_method``). Per ``(layer, position)``:

- ``contrast_control     = mean(act_ctrl_GT_1)     - mean(act_ctrl_GT_0)``
- ``contrast_experiment  = mean(act_exp_GT_1)      - mean(act_exp_GT_0)``
- ``cos_sim[layer, pos]  = cosine(contrast_control, contrast_experiment)``

A cosine of ``+1`` means the two conditions point the contrast direction the
same way (no rotation); ``-1`` means the experiment fully flipped the direction
(maximum rotation). The rotation-metric plugin in
:mod:`src.analysis.rotation_plugins` reduces the per-layer curve at one
position to a scalar.
"""

from __future__ import annotations

import numpy as np


def _cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    """Numerically safe cosine similarity for 1-D vectors."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def compute_cos_sim_matrix(
    act_ctrl_gt0: np.ndarray,
    act_ctrl_gt1: np.ndarray,
    act_exp_gt0: np.ndarray,
    act_exp_gt1: np.ndarray,
) -> np.ndarray:
    """
    Compute the per-``(layer, position)`` cosine matrix between the control
    and experiment contrast directions.

    Args:
        act_ctrl_gt0: ``(k, n_positions, n_layers, d_model)`` control / GT_0.
        act_ctrl_gt1: ``(k, n_positions, n_layers, d_model)`` control / GT_1.
        act_exp_gt0:  ``(k, n_positions, n_layers, d_model)`` experiment / GT_0.
        act_exp_gt1:  ``(k, n_positions, n_layers, d_model)`` experiment / GT_1.
            ``k`` may differ across the four arrays only across calls — within a
            single call, all four are expected to be the same balanced size.

    Returns:
        ``cos[n_layers, n_positions]`` array of cosine similarities.

    Raises:
        ValueError: If shapes disagree on ``n_positions``, ``n_layers``, or
            ``d_model``.
    """
    shapes = {
        "ctrl_gt0": act_ctrl_gt0.shape,
        "ctrl_gt1": act_ctrl_gt1.shape,
        "exp_gt0": act_exp_gt0.shape,
        "exp_gt1": act_exp_gt1.shape,
    }
    if len({s[1:] for s in shapes.values()}) != 1:
        raise ValueError(f"Activation tensors disagree on (n_pos, n_layers, d_model): {shapes}")

    _, n_positions, n_layers, _ = act_ctrl_gt0.shape

    # Mean over the balanced-set dimension first (cheap; reduces memory).
    mean_ctrl_gt0 = act_ctrl_gt0.mean(axis=0)  # (n_pos, n_layers, d)
    mean_ctrl_gt1 = act_ctrl_gt1.mean(axis=0)
    mean_exp_gt0 = act_exp_gt0.mean(axis=0)
    mean_exp_gt1 = act_exp_gt1.mean(axis=0)

    contrast_ctrl = mean_ctrl_gt1 - mean_ctrl_gt0  # (n_pos, n_layers, d)
    contrast_exp = mean_exp_gt1 - mean_exp_gt0

    cos = np.zeros((n_layers, n_positions), dtype=np.float64)
    for L in range(n_layers):
        for p in range(n_positions):
            cos[L, p] = _cosine(contrast_ctrl[p, L], contrast_exp[p, L])
    return cos
