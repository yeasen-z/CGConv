"""Hook-based activation and effective-FLOPs statistics."""

from collections import defaultdict
from typing import Iterable, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _merge_inputs(inputs: Union[Tensor, Sequence[Tensor]]) -> Tensor:
    if isinstance(inputs, Tensor):
        tensors = [inputs]
    else:
        tensors = list(inputs)
    if not tensors or any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("FLOPs analysis expects one or more NCHW tensors")
    if not all(tensor.shape[1:] == tensors[0].shape[1:] for tensor in tensors):
        raise ValueError("non-batch activation dimensions must match")
    return torch.cat(tensors, dim=0)


def remove_hooks(hooks) -> None:
    for hook in hooks:
        hook.remove()


class LayersZeroBatchRatioMeter:
    """Collect active element/window ratios for selected layers.

    ``f_type='Sponge'`` uses the CGConv centre rule. ``f_type='Sponge0'``
    retains a window whenever any value in that window is non-zero. The legacy
    names are retained because the analysis notebook uses them.
    """

    def __init__(self, f_type: str = "Sponge"):
        normalised = f_type.lower()
        if normalised in {"sponge", "center", "centre"}:
            self.centre_only = True
        elif normalised in {"sponge0", "any", "nonzero_window"}:
            self.centre_only = False
        else:
            raise ValueError(f"unknown FLOPs analysis type: {f_type!r}")

        self.k1_s1_fired = defaultdict(list)
        self.k1_s1_fired_perc = defaultdict(list)
        self.k3_s1_fired = defaultdict(list)
        self.k3_s1_fired_perc = defaultdict(list)
        self.k3_s2_fired = defaultdict(list)
        self.k3_s2_fired_perc = defaultdict(list)

    def register_output_stats(self, name, inputs) -> None:
        activation = _merge_inputs(inputs).detach()
        fired = activation.count_nonzero()
        self.k1_s1_fired[name].append(float(fired))
        self.k1_s1_fired_perc[name].append(float(fired / activation.numel()))

        for stride, fired_store, ratio_store in (
            (1, self.k3_s1_fired, self.k3_s1_fired_perc),
            (2, self.k3_s2_fired, self.k3_s2_fired_perc),
        ):
            windows = F.unfold(
                activation, kernel_size=3, dilation=1, padding=1, stride=stride
            ).view(activation.shape[0], activation.shape[1], 9, -1)
            if self.centre_only:
                active = windows[:, :, 4, :].ne(0)
            else:
                active = windows.ne(0).any(dim=2)
            active_count = active.count_nonzero()
            fired_store[name].append(float(active_count))
            ratio_store[name].append(float(active_count / active.numel()))

    def avg_fired(self) -> None:
        stores = (
            self.k1_s1_fired,
            self.k1_s1_fired_perc,
            self.k3_s1_fired,
            self.k3_s1_fired_perc,
            self.k3_s2_fired,
            self.k3_s2_fired_perc,
        )
        for store in stores:
            for key, values in store.items():
                store[key] = float(np.mean(values))


def flops_ratio(
    model,
    inference_func,
    layer_list: Iterable[str] = (),
    f_type: str = "Sponge",
):
    """Run inference and collect activation ratios for named model layers."""
    requested = set(layer_list)
    selected = {
        name: layer for name, layer in model.named_modules() if name in requested
    }
    missing = requested.difference(selected)
    if missing:
        raise ValueError(f"unknown layer names: {sorted(missing)}")

    stats = LayersZeroBatchRatioMeter(f_type=f_type)
    hooks = []

    def hook_fn(name):
        def register_stats_hook(_module, inputs, _output):
            stats.register_output_stats(name, inputs)

        return register_stats_hook

    for name, module in selected.items():
        hooks.append(module.register_forward_hook(hook_fn(name)))

    try:
        inference_func(model)
    finally:
        remove_hooks(hooks)
    stats.avg_fired()
    return stats
