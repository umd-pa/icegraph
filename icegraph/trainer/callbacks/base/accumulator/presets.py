# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Iterator
from functools import partial

import torch
from torch import Tensor

from .accumulator import Accumulator

__all__ = ["dense_histogram_accumulator", "sparse_histogram_accumulator"]


# ops

def _sum_combiner(t1: Tensor, t2: Tensor) -> None:
    t1.add_(t2)

def _sparse_histogram_combiner(t1: Tensor, t2: Tensor, *, ndim: int) -> Tensor:
    # unfortunately cannot be done in-place :(
    # concatenate
    combined = torch.cat([t1, t2], dim=0)

    # grab keys and counts
    keys    = combined[:, :ndim + 1]
    counts  = combined[:,  ndim + 1].to(t1.dtype)

    # determine unique keys and their group id (inv)
    unique_keys, inv = torch.unique(keys, dim=0, return_inverse=True)

    # initialize empty tensor to scatter_add to
    combined_counts = torch.zeros(unique_keys.size(0), device=t1.device, dtype=t1.dtype)
    combined_counts.scatter_add_(0, inv, counts)

    # update t1 with combined and deduped
    return torch.cat([unique_keys, combined_counts.unsqueeze(1)], dim=1)

def _first_dim_enumerator(t: Tensor) -> Iterator[tuple[int, Tensor]]:
    for i, item in enumerate(t):
        item: Tensor
        yield i, item

def _first_item_enumerator(t: Tensor) -> Iterator[tuple[int, Tensor]]:
    indices = t[:, 0].to(torch.long)

    # Sort once so equal indices become contiguous
    order = torch.argsort(indices, stable=True)
    t_sorted = t.index_select(0, order)
    indices_sorted = indices.index_select(0, order)

    # find run lengths for each index
    unique, counts = torch.unique_consecutive(indices_sorted, return_counts=True)

    # Split the remaining columns into per-index chunks
    chunks = torch.split(t_sorted[:, 1:], counts.tolist())

    for index, chunk in zip(unique, chunks):
        yield int(index.item()), chunk

# presets

def dense_histogram_accumulator(
        size: tuple[int, ...], device: torch.device, dtype: torch.dtype = torch.long
) -> Accumulator:
    # build dense accumulator
    tensor = torch.zeros(size, device=device, dtype=dtype)

    return Accumulator(tensor, _sum_combiner, _first_dim_enumerator)

def sparse_histogram_accumulator(
        ndim: int, device: torch.device, dtype: torch.dtype = torch.long
) -> Accumulator:
    # entries have shape (index, bin_0, ..., bin_{ndim-1}, count), thus
    if ndim < 1:
        raise ValueError("ndim for sparse histogram accumulators must be at least 1.")

    tensor = torch.empty(0, ndim + 2, device=device, dtype=dtype)

    combiner = partial(_sparse_histogram_combiner, ndim=ndim)

    return Accumulator(tensor, combiner, _first_item_enumerator)
