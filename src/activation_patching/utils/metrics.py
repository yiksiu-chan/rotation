"""Behavior scores and streaming contrast-direction statistics for patching."""
from __future__ import annotations
import numpy as np
from src.analysis.metrics import compute_cos_sim_matrix


def honesty(logits, labels, id0, id1):
    # returns the correct minus incorrect logits
    import torch
    difference = logits[:, id1].float() - logits[:, id0].float()
    labels = torch.as_tensor(labels, device=logits.device)
    return torch.where(labels == 1, difference, -difference).detach().cpu()


def argmax_matches_truth(logits, labels, id0, id1):
    """Per-row: is the single most likely token (argmax over the FULL vocabulary)
    exactly the ground-truth-correct answer word?

    Stricter than honesty()'s sign check, which only compares id0 against id1
    and says nothing about whether either is actually competitive against the
    rest of the vocabulary.
    """
    import torch
    top1 = logits.argmax(dim=-1)
    labels = torch.as_tensor(labels, device=logits.device)
    id0_t = torch.full_like(top1, id0)
    id1_t = torch.full_like(top1, id1)
    truth_id = torch.where(labels == 1, id1_t, id0_t)
    return (top1 == truth_id).detach().cpu()


def select_rows(control, experiment, policy):
    """Preserve the rebuttal's zero-matches fallback and expose it in results."""
    import torch
    # Keep statements that were honest in control and dishonest in experiment.
    # The same baseline mask is used for all patches and both metrics.
    mask = (control > 0) & (experiment < 0) if policy == 'logitdiff' else torch.ones_like(control, dtype=torch.bool)
    fallback = not bool(mask.any())
    if fallback:
        # Historical convention, exposed as filter_fallback in saved results.
        # With this fallback (or policy='none'), flip_rate is a final-preference
        # fraction and is not necessarily a fraction that CHANGED preference.
        mask = torch.ones_like(mask)
    return mask, fallback


def behavior_scores(control, experiment, patched, mask, direction):
    ctrl = float(control[mask].mean())
    exp = float(experiment[mask].mean())
    value = float(patched[mask].mean())
    start, end = (exp, ctrl) if direction == 'denoise' else (ctrl, exp)
    gap = end - start
    effect = (value - start) / gap if abs(gap) > 1e-6 else None
    # count the percentage of sign flip after patching
    # mask: flips wrong -> right and flips right -> wrong.
    flips = patched[mask] > 0 if direction == 'denoise' else patched[mask] < 0
    return {'logit_diff': value, 'logit_shift': value - start,
            'recovery': effect, 'flip_rate': float(flips.float().mean())}


def early_to_late_drop(curve):
    """Largest nonnegative cosine drop from the first half to the second half.

    Split at n_layers // 2: for 32 layers, early is 0 to 15 and late is 16 to 31.
    For an odd layer count, the second half includes the extra layer.
    """
    values = np.asarray(curve, dtype=np.float64)
    if values.size < 2:
        return 0.0
    midpoint = values.size // 2
    early = values[:midpoint]
    late = values[midpoint:]
    # Only cross-half pairs are eligible; drops entirely within a half do not count.
    return max(0.0, float(np.max(early) - np.min(late)))


def best_rotation(cosine):
    # Choose a TOKEN POSITION from the baseline matrix [layer, token].
    values = [early_to_late_drop(cosine[:, p]) for p in range(cosine.shape[1])]
    position = int(np.nanargmax(values))
    return values[position], position


class ContrastAccumulator:
    """Accumulate per-GT means on CPU, independent of GPU microbatch size.

    Inputs retain the shared generation cache's float16 precision. Only sums
    and counts are retained, rather than every row's hidden states.
    """
    def __init__(self):
        self.sums = {}
        self.counts = {0: 0, 1: 0}

    def add(self, cache, labels, mask):
        values = cache.detach().cpu().permute(1, 2, 0, 3).float().numpy()
        labels = np.asarray(labels)
        keep = np.asarray(mask, dtype=bool)
        for gt in (0, 1):
            selected = values[(labels == gt) & keep]
            if len(selected):
                total = selected.sum(axis=0, dtype=np.float64)
                self.sums[gt] = self.sums.get(gt, 0) + total
                self.counts[gt] += len(selected)

    def means(self):
        # Separate class means; each row has equal weight within its GT class.
        # Unlike the mean PATCH SOURCE, these are filtered, class-specific means
        # used only to MEASURE the residual-stream contrast direction.
        if not all(self.counts.values()):
            raise ValueError('Rotation requires kept examples in both GT classes; use more rows or --filter none.')
        return [(self.sums[gt] / self.counts[gt]).astype(np.float32)[None] for gt in (0, 1)]


def contrast_cosine(control, experiment):
    # At every layer/token: d = mean(hidden | GT1) - mean(hidden | GT0).
    # Compare d_control and d_experiment using cosine similarity, not individual
    # example vectors. Hidden states are block outputs collected in generation.
    return compute_cos_sim_matrix(*control.means(), *experiment.means())
