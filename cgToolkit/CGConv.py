import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import torch
import torch.nn as nn

class CGConv2d(nn.Conv2d):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor],centre=True):
        N, C, H, W = input.shape
        K_h, K_w = self.kernel_size
        
        in_unfolded=F.unfold(input, self.kernel_size, dilation=self.dilation, padding=self.padding,stride=self.stride)

        in_unfolded_k = in_unfolded.view(N, self.in_channels, K_h, K_w, -1)
        in_unfolded_NCLKK = in_unfolded_k.permute(0, 1, 4, 2, 3)
        
        Hout = int((H+self.padding[0]*2-self.dilation[0]*(K_h-1)-1)/self.stride[0])+1
        Wout = int((W+self.padding[1]*2-self.dilation[1]*(K_w-1)-1)/self.stride[0])+1
        in_unfolded_NCHinHoutKK = in_unfolded_NCLKK.view(N, self.in_channels, Hout, Wout, K_h, K_w)
        
        for_mask = in_unfolded_NCHinHoutKK.clone()

        if centre:
            center_index_H = K_h // 2
            center_index_W = K_w // 2
            center_values = for_mask[:,:,:,:,center_index_H, center_index_W]
            center_values[center_values!=0]=1
            mask = center_values.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, K_h, K_w)
        else:
            ifall_zero = for_mask[:,:,:,:,:,:]
            ifall_zero[ifall_zero!=0]=1
            ifall_zero_sum = torch.sum(ifall_zero, dim=(-2, -1))
            ifall_zero_sum[ifall_zero_sum!=0]=1
            mask = ifall_zero_sum.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, K_h, K_w)
        
        in_unfolded_NCHinHoutKK = in_unfolded_NCHinHoutKK * mask
        
        x_unfold = in_unfolded_NCHinHoutKK.permute(0, 1, 2, 4, 3, 5).contiguous()
        x_unfold = x_unfold.view(N, C, Hout * K_h, Wout * K_w)
        
        out = F.conv2d(x_unfold, weight, bias, stride=(K_h,K_w), padding=0, dilation=1, groups=self.groups)

        return out
