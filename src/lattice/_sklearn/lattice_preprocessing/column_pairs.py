"""Diversify column pairs seen by Lattice Small's group size-three embedder."""

from math import gcd

import numpy as np


def pair_keys(columns, original_width):
    columns = columns.astype(np.int64, copy=False)
    keys = []
    for offset in (1, 2, 3):
        shifted = np.roll(columns, -offset)
        keys.append(np.minimum(columns, shifted) * original_width + np.maximum(columns, shifted))
    return np.unique(np.concatenate(keys))


def distinct_pairs(permutations, original_width):
    return np.unique(np.concatenate([pair_keys(columns, original_width) for columns in permutations])).size


def reduce_pair_reuse(permutations, original_width):
    width = len(permutations[0])
    if len(permutations) != 8 or width < 8:
        return permutations
    offsets = np.arange(width)
    strides = [step for step in range(1, (width + 1) // 2) if gcd(step, width) == 1]
    strides = [strides[index] for index in np.linspace(0, len(strides) - 1, min(48, len(strides)), dtype=int)]
    selected = list(permutations[:4])
    seen = np.sort(np.concatenate([pair_keys(columns, original_width) for columns in selected]))
    for columns in permutations[4:]:
        lowest_reuse = None
        for stride in strides:
            trial = columns[(offsets * stride) % width]
            keys = pair_keys(trial, original_width)
            reuse = (seen.searchsorted(keys, side="right") - seen.searchsorted(keys)).sum()
            if lowest_reuse is None or reuse < lowest_reuse:
                best, best_keys, lowest_reuse = trial, keys, reuse
        selected.append(best)
        seen = np.sort(np.concatenate([seen, best_keys]))
    if distinct_pairs(selected, original_width) <= distinct_pairs(permutations, original_width):
        return permutations
    return selected
