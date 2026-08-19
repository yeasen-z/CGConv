"""Activation sparsity regularisation used to train CGConv models."""

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _merge_inputs(inputs: Union[Tensor, Sequence[Tensor]]) -> Tensor:
    """Return one NCHW tensor from a module input or activation tensor."""
    if isinstance(inputs, Tensor):
        tensors = [inputs]
    else:
        tensors = list(inputs)

    if not tensors:
        raise ValueError("cannot calculate a sparsity loss without activations")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("the sparsity loss expects NCHW activations")
    if not all(tensor.shape[1:] == tensors[0].shape[1:] for tensor in tensors):
        raise ValueError("non-batch activation dimensions must match")
    return torch.cat(tensors, dim=0)


@dataclass
class SpongeMeter:
    """Collect differentiable sparsity losses and detached diagnostics."""

    args: object
    loss: list = field(default_factory=list)
    fired_perc: list = field(default_factory=list)
    fired: list = field(default_factory=list)
    l2: list = field(default_factory=list)

    def register_output_stats(
        self, inputs: Union[Tensor, Sequence[Tensor]]
    ) -> None:
        activation = _merge_inputs(inputs)

        if self.args.sponge_criterion == "l0":
            sigma = self.args.sponge_sigma
            approx_l0 = (activation.square() / (activation.square() + sigma)).mean()
        elif self.args.sponge_criterion == "l2":
            channels = activation.shape[1]
            windows = F.unfold(
                activation, kernel_size=3, dilation=1, padding=1, stride=1
            ).view(activation.shape[0], channels, 3, 3, -1)
            approx_l0 = windows.var(dim=(2, 3), unbiased=False).mean()
        else:
            raise ValueError(
                f"unknown sponge criterion: {self.args.sponge_criterion!r}"
            )

        detached = activation.detach()
        fired = detached.count_nonzero()
        self.loss.append(approx_l0)
        self.fired.append(fired)
        self.fired_perc.append(fired / detached.numel())
        self.l2.append(detached.norm(2))


def register_hooks(leaf_nodes: Iterable[nn.Module], hook):
    """Register a forward hook on every selected module."""
    return [node.register_forward_hook(hook) for node in leaf_nodes]


def remove_hooks(hooks) -> None:
    for hook in hooks:
        hook.remove()


def do_sponge_loss(model, inputs, victim_leaf_nodes, args):
    """Run one forward pass and return its sparsity penalty and output.

    ``victim_leaf_nodes`` should normally be the model's ``CGConv2d`` layers.
    Their forward inputs are exactly the activations inspected by CGConv.
    """
    sponge_stats = SpongeMeter(args)

    def register_stats_hook(_module, module_inputs, _output):
        sponge_stats.register_output_stats(module_inputs)

    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)
    try:
        outputs = model(inputs)
    finally:
        remove_hooks(hooks)

    if not sponge_stats.loss:
        raise ValueError(
            "no activations were collected; select a model containing CGConv2d "
            "layers or disable --cgc"
        )

    output_device = outputs.device
    sponge_loss = torch.stack(
        [layer_loss.to(output_device) for layer_loss in sponge_stats.loss]
    ).mean()
    sponge_loss = sponge_loss * args.sponge_lb

    fired = sum(float(value) for value in sponge_stats.fired)
    fired_perc = sum(float(value) for value in sponge_stats.fired_perc)
    fired_perc /= len(sponge_stats.fired_perc)
    l2 = sum(float(value) for value in sponge_stats.l2)
    diagnostics = (float(sponge_loss.detach()), fired, fired_perc, l2)
    return sponge_loss, outputs, diagnostics


def compute_sponge_loss(model, inputs, victim_leaf_nodes, args):
    """Compatibility wrapper returning only the penalty and diagnostics."""
    sponge_loss, _, diagnostics = do_sponge_loss(
        model, inputs, victim_leaf_nodes, args
    )
    return sponge_loss, {
        "sponge_loss": float(sponge_loss.detach()),
        "sponge_stats": diagnostics,
    }
