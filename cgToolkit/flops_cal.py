import torch
from collections import defaultdict
import numpy as np
import torch.nn.functional as F

def remove_hooks(hooks):
    """
    Remove hooks from a model.
    :param hooks: an Iterable containing hooks to be removed.
    """
    for hook in hooks:
        hook.remove()

class LayersZeroBatchRatioMeter:
    def __init__(self, f_type='Sponge'):
        self.k1_s1_fired = defaultdict(list)
        self.k1_s1_fired_perc = defaultdict(list)

        self.k3_s1_fired = defaultdict(list)
        self.k3_s1_fired_perc = defaultdict(list)

        self.k3_s2_fired = defaultdict(list)
        self.k3_s2_fired_perc = defaultdict(list)
        
        self.f_type = f_type
      
    def register_output_stats(self, name, input):
        if self.f_type == 'Sponge':
            self.calculate_sponge(name, input)
        elif self.f_type == 'Sponge0':
            self.calculate_sponge0(name, input)
                
    def calculate_sponge(self, name, input):
        if all(tensor.shape[1:] == input[0].shape[1:] for tensor in input):
            combined_input = torch.cat(input, dim=0)  # 沿 Batch 维度拼接
        else:
            raise ValueError("非 Batch 维度不一致，无法沿 Batch 维度拼接！")
        input_test=combined_input.clone()
        N,C,H,W=combined_input.shape
        
        # k1_s1_fired
        k1_s1_fired = input_test.norm(0)
        k1_s1_fired_perc = k1_s1_fired / (N*C*H*W)

        self.k1_s1_fired[name].append(k1_s1_fired.item())
        self.k1_s1_fired_perc[name].append(k1_s1_fired_perc.item())

        # k3_s1_fired
        K_h,K_w,K=3,3,3
        padding = (K-1)//2
        
        in_unfolded=F.unfold(input_test, kernel_size=K, dilation=1, padding=padding,stride=1)
        in_unfolded_k = in_unfolded.view(N, C, K_h, K_w, -1)
        in_unfolded_NCLKK = in_unfolded_k.permute(0, 1, 4, 2, 3)
        
        Hout = int(H+padding*2-K_h)+1
        Wout = int(W+padding*2-K_w)+1
        in_unfolded_NCHinHoutKK = in_unfolded_NCLKK.view(N, C, Hout, Wout, K_h, K_w)
        
        for_mask = in_unfolded_NCHinHoutKK.clone()
        
        center_index_H = K_h // 2
        center_index_W = K_w // 2
        center_values = for_mask[:,:,:,:,center_index_H, center_index_W]
        center_values[center_values!=0]=1
        mask=center_values

        k3_s1_fired = mask.norm(0)
        k3_s1_fired_perc = k3_s1_fired / (N*C*H*W)

        self.k3_s1_fired[name].append(k3_s1_fired.item())
        self.k3_s1_fired_perc[name].append(k3_s1_fired_perc.item())

        # k3_s2_fired
        K_h,K_w,K=3,3,3
        padding = (K-1)//2

        in_unfolded=F.unfold(input_test, kernel_size=K, dilation=1, padding=padding,stride=2)
        in_unfolded_k = in_unfolded.view(N, C, K_h, K_w, -1)
        in_unfolded_NCLKK = in_unfolded_k.permute(0, 1, 4, 2, 3)

        Hout = int((H+padding*2-K_h)/2)+1
        Wout = int((W+padding*2-K_w)/2)+1
        in_unfolded_NCHinHoutKK = in_unfolded_NCLKK.view(N, C, Hout, Wout, K_h, K_w)

        for_mask = in_unfolded_NCHinHoutKK.clone()

        center_index_H = K_h // 2
        center_index_W = K_w // 2
        center_values = for_mask[:,:,:,:,center_index_H, center_index_W]
        center_values[center_values!=0]=1
        mask=center_values

        k3_s2_fired = mask.norm(0)
        k3_s2_fired_perc = k3_s2_fired / ((N*C*H*W)/4)

        self.k3_s2_fired[name].append(k3_s2_fired.item())
        self.k3_s2_fired_perc[name].append(k3_s2_fired_perc.item())

    def calculate_sponge0(self, name, input):
        if all(tensor.shape[1:] == input[0].shape[1:] for tensor in input):
            combined_input = torch.cat(input, dim=0)  # 沿 Batch 维度拼接
        else:
            raise ValueError("非 Batch 维度不一致，无法沿 Batch 维度拼接！")
        input_test=combined_input.clone()
        N,C,H,W=combined_input.shape
        
        # k1_s1_fired
        k1_s1_fired = input_test.norm(0)
        k1_s1_fired_perc = k1_s1_fired / (N*C*H*W)

        self.k1_s1_fired[name].append(k1_s1_fired.item())
        self.k1_s1_fired_perc[name].append(k1_s1_fired_perc.item())

        # k3_s1_fired
        K_h,K_w,K=3,3,3
        padding = (K-1)//2
        
        in_unfolded=F.unfold(input_test, kernel_size=K, dilation=1, padding=padding,stride=1)
        in_unfolded_k = in_unfolded.view(N, C, K_h, K_w, -1)
        in_unfolded_NCLKK = in_unfolded_k.permute(0, 1, 4, 2, 3)
        
        Hout = int(H+padding*2-K_h)+1
        Wout = int(W+padding*2-K_w)+1
        in_unfolded_NCHinHoutKK = in_unfolded_NCLKK.view(N, C, Hout, Wout, K_h, K_w)
        
        for_mask = in_unfolded_NCHinHoutKK.clone()
        
        ifall_zero = for_mask[:,:,:,:,:,:]
        ifall_zero[ifall_zero!=0]=1
        ifall_zero_sum = torch.sum(ifall_zero, dim=(-2, -1))
        ifall_zero_sum[ifall_zero_sum!=0]=1
        mask=ifall_zero_sum

        k3_s1_fired = mask.norm(0)
        k3_s1_fired_perc = k3_s1_fired / (N*C*H*W)

        self.k3_s1_fired[name].append(k3_s1_fired.item())
        self.k3_s1_fired_perc[name].append(k3_s1_fired_perc.item())

        # k3_s2_fired
        K_h,K_w,K=3,3,3
        padding = (K-1)//2

        in_unfolded=F.unfold(input_test, kernel_size=K, dilation=1, padding=padding,stride=2)
        in_unfolded_k = in_unfolded.view(N, C, K_h, K_w, -1)
        in_unfolded_NCLKK = in_unfolded_k.permute(0, 1, 4, 2, 3)

        Hout = int((H+padding*2-K_h)/2)+1
        Wout = int((W+padding*2-K_w)/2)+1
        in_unfolded_NCHinHoutKK = in_unfolded_NCLKK.view(N, C, Hout, Wout, K_h, K_w)

        for_mask = in_unfolded_NCHinHoutKK.clone()

        ifall_zero = for_mask[:,:,:,:,:,:]
        ifall_zero[ifall_zero!=0]=1
        ifall_zero_sum = torch.sum(ifall_zero, dim=(-2, -1))
        ifall_zero_sum[ifall_zero_sum!=0]=1
        mask=ifall_zero_sum

        k3_s2_fired = mask.norm(0)
        k3_s2_fired_perc = k3_s2_fired / ((N*C*H*W)/4)

        self.k3_s2_fired[name].append(k3_s2_fired.item())
        self.k3_s2_fired_perc[name].append(k3_s2_fired_perc.item())
        
    def avg_fired(self):
        for key in self.k1_s1_fired_perc.keys():
            self.k1_s1_fired_perc[key] = np.mean(self.k1_s1_fired_perc[key])
            self.k3_s1_fired_perc[key] = np.mean(self.k3_s1_fired_perc[key])
            self.k3_s2_fired_perc[key] = np.mean(self.k3_s2_fired_perc[key])
            


def flops_ratio(model, inference_func, layer_list=[], f_type='Sponge'):
    hooks = []

    common_target_layer = []
    common_target_name = []
    for name, layer in model.named_modules():
        if name in layer_list:
            common_target_layer.append(layer)
            common_target_name.append(name)

    print(len(common_target_layer))
    
    stats = LayersZeroBatchRatioMeter(f_type=f_type)
    
    def hook_fn(name):
        def register_stats_hook(model, input, output):
            stats.register_output_stats(name, input)

        return register_stats_hook
    
    ids = defaultdict(int)

    for i, module in enumerate(common_target_layer):
        module_name = str(module).split('(')[0]
        hook = module.register_forward_hook(hook_fn(f'{module_name}-{ids[module_name]}'))
        ids[module_name] += 1
        hooks.append(hook)

    inference_func(model)
        
    stats.avg_fired()

    remove_hooks(hooks)
    return stats