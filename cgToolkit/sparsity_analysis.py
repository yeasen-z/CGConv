"""Analyse regular and irregular zeros in intermediate activations."""

from collections import defaultdict
from typing import Iterable, Sequence, Union

import numpy as np
import torch
from torch import Tensor


def _merge_inputs(inputs: Union[Tensor, Sequence[Tensor]]) -> Tensor:
    if isinstance(inputs, Tensor):
        tensors = [inputs]
    else:
        tensors = list(inputs)
    if not tensors or any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("sparsity analysis expects one or more NCHW tensors")
    if not all(tensor.shape[1:] == tensors[0].shape[1:] for tensor in tensors):
        raise ValueError("non-batch activation dimensions must match")
    return torch.cat(tensors, dim=0)


def remove_hooks(hooks) -> None:
    for hook in hooks:
        hook.remove()


class LayersSparsityMeter:
    """Estimate how many zeros form channel- or position-level patterns."""

    def __init__(self, th_ratio: float = 0.97):
        if not 0 <= th_ratio <= 1:
            raise ValueError("th_ratio must be between 0 and 1")
        self.th_ratio = th_ratio
        self.irregular_zero = defaultdict(list)
        self.irregular_zero_perc = defaultdict(list)
        self.regular_zero = defaultdict(list)
        self.regular_zero_perc = defaultdict(list)
        self.total_zero_num = defaultdict(list)
        self.total_zero_perc = defaultdict(list)
        self.HW_size = defaultdict(list)
        self.CH_len = defaultdict(list)

    def register_input_stats(self, name, inputs) -> None:
        activation = _merge_inputs(inputs).detach()
        batch, channels, height, width = activation.shape
        spatial_size = height * width

        self.HW_size[name].append(spatial_size)
        self.CH_len[name].append(channels)

        is_zero = activation.eq(0)
        zero_counts = is_zero.flatten(1).sum(dim=1)
        self.total_zero_num[name].append(float(zero_counts.float().mean()))
        self.total_zero_perc[name].append(
            float(zero_counts.sum() / (batch * channels * spatial_size))
        )

        channel_zero_ratio = is_zero.sum(dim=(2, 3)) / spatial_size
        sparse_channels = channel_zero_ratio >= self.th_ratio
        sparse_channel_counts = sparse_channels.sum(dim=1)

        position_zero_ratio = is_zero.sum(dim=1) / channels
        sparse_positions = position_zero_ratio >= self.th_ratio
        sparse_position_counts = sparse_positions.sum(dim=(1, 2))

        # This is a pattern-coverage estimate. Clamp it to the number of actual
        # zeros so threshold overlap cannot create negative irregular counts.
        regular_counts = sparse_channel_counts * spatial_size
        regular_counts += sparse_position_counts * (
            channels - sparse_channel_counts
        )
        regular_counts = torch.minimum(regular_counts, zero_counts)
        irregular_counts = zero_counts - regular_counts
        safe_zero_counts = zero_counts.clamp_min(1)

        self.regular_zero[name].append(float(regular_counts.float().mean()))
        self.irregular_zero[name].append(float(irregular_counts.float().mean()))
        self.regular_zero_perc[name].append(
            float((regular_counts / safe_zero_counts).float().mean())
        )
        self.irregular_zero_perc[name].append(
            float((irregular_counts / safe_zero_counts).float().mean())
        )

    def avg_sparsity(self) -> None:
        stores = (
            self.total_zero_num,
            self.total_zero_perc,
            self.regular_zero,
            self.regular_zero_perc,
            self.irregular_zero,
            self.irregular_zero_perc,
            self.HW_size,
            self.CH_len,
        )
        for store in stores:
            for key, values in store.items():
                store[key] = float(np.mean(values))


def get_sparsity(
    model,
    inference_func,
    layer_list: Iterable[str] = (),
    th_ratio: float = 0.97,
):
    """Run inference and collect sparsity statistics for named Conv2d layers."""
    requested = set(layer_list)
    selected = {
        name: layer
        for name, layer in model.named_modules()
        if name in requested and isinstance(layer, torch.nn.Conv2d)
    }
    missing = requested.difference(selected)
    if missing:
        raise ValueError(f"unknown Conv2d layer names: {sorted(missing)}")

    stats = LayersSparsityMeter(th_ratio=th_ratio)
    hooks = []

    def hook_fn(name):
        def register_stats_hook(_module, inputs, _output):
            stats.register_input_stats(name, inputs)

        return register_stats_hook

    for name, module in selected.items():
        hooks.append(module.register_forward_hook(hook_fn(name)))

    try:
        inference_func(model)
    finally:
        remove_hooks(hooks)
    stats.avg_sparsity()
    return stats
