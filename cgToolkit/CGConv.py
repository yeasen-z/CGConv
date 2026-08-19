"""PyTorch implementation of Center Glance Convolution."""

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CGConv2d(nn.Conv2d):
    """A convolution that drops a window when its centre activation is zero.

    For every input channel and output position, the convolution window is
    retained only when its centre value is non-zero. This produces a regular,
    window-level sparsity pattern that can be exploited by the HLS prototype.

    The PyTorch implementation is a functional reference. It still executes a
    dense convolution after masking, so it should not be expected to run faster
    than :class:`torch.nn.Conv2d` on a GPU.
    """

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        centre: bool = True,
    ) -> Tensor:
        if input.ndim != 4:
            raise ValueError(f"CGConv2d expects an NCHW tensor, got {input.shape}")

        if self.padding_mode != "zeros":
            input = F.pad(
                input,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
            padding = (0, 0)
        else:
            padding = self.padding

        batch_size, channels, height, width = input.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        padding_h, padding_w = padding

        output_h = (
            height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        output_w = (
            width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1
        if output_h <= 0 or output_w <= 0:
            raise ValueError("kernel size is larger than the padded input")

        unfolded = F.unfold(
            input,
            self.kernel_size,
            dilation=self.dilation,
            padding=padding,
            stride=self.stride,
        )
        windows = unfolded.view(
            batch_size, channels, kernel_h, kernel_w, output_h, output_w
        ).permute(0, 1, 4, 5, 2, 3)

        if centre:
            centre_values = windows[..., kernel_h // 2, kernel_w // 2]
            active = centre_values.ne(0)
        else:
            active = windows.ne(0).any(dim=(-2, -1))

        masked_windows = windows * active[..., None, None].to(windows.dtype)

        # Tile independent windows into an image and use a kernel-sized stride.
        # This is equivalent to multiplying each retained unfolded window by
        # the original convolution weights.
        tiled = masked_windows.permute(0, 1, 2, 4, 3, 5).contiguous()
        tiled = tiled.view(
            batch_size,
            channels,
            output_h * kernel_h,
            output_w * kernel_w,
        )
        return F.conv2d(
            tiled,
            weight,
            bias,
            stride=(kernel_h, kernel_w),
            padding=0,
            dilation=1,
            groups=self.groups,
        )
