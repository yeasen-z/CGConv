import torch
import torch.nn.functional as F


class SpongeMeter:
    def __init__(self, args):
        self.loss = []
        self.fired_perc = []
        self.fired = []
        self.l2 = []
        self.src_loss = []
        self.size = 0

        self.sigma = args.sponge_sigma
        self.args = args

    def register_output_stats(self, input):
        if all(tensor.shape[1:] == input[0].shape[1:] for tensor in input):
            combined_input = torch.cat(input, dim=0)  # 沿 Batch 维度拼接
        else:
            raise ValueError("非 Batch 维度不一致，无法沿 Batch 维度拼接！")
        inp=combined_input.clone()
        # inp = input.clone()

        if self.args.sponge_criterion == 'l0':
            approx_norm_0 = torch.sum(inp ** 2 / (inp ** 2 + self.sigma)) / inp.numel()
        elif self.args.sponge_criterion == 'l2':
            N, C, H, W = inp.shape
            in_unfolded=F.unfold(inp, kernel_size=3, dilation=1, padding=1,stride=1)
            in_unfolded_k = in_unfolded.view(N, C, 3, 3, -1)
            in_unfolded_NCLKK = in_unfolded_k.permute(0, 1, 4, 2, 3)
            in_unfolded_var = torch.var(in_unfolded_NCLKK, dim=(-2, -1))
            var_sum = torch.sum(in_unfolded_var)
            approx_norm_0=var_sum/inp.numel()
        else:
            raise ValueError('Invalid sponge criterion loss')

        fired = inp.detach().norm(0)
        fired_perc = fired / inp.detach().numel()

        self.loss.append(approx_norm_0)
        self.fired.append(fired)
        self.fired_perc.append(fired_perc)
        self.l2.append(inp.detach().norm(2))
        self.size += 1

    def register_stats(self, stats):
        sponge_loss, src_loss, fired, fired_perc, l2 = stats
        self.loss.append(sponge_loss)
        self.src_loss.append(src_loss)
        self.fired.append(fired)
        self.fired_perc.append(fired_perc)
        self.l2.append(l2)
        self.size += 1


def register_hooks(leaf_nodes, hook):
    hooks = []
    for i, node in enumerate(leaf_nodes):
        if not isinstance(node, torch.nn.modules.dropout.Dropout):
            hooks.append(node.register_forward_hook(hook))
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def do_sponge_loss(model, x, victim_leaf_nodes, args):
    sponge_stats = SpongeMeter(args)

    def register_stats_hook(model, input, output):
        sponge_stats.register_output_stats(input)

    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)

    outputs = model(x)

    sponge_loss = fired_perc = fired = l2 = 0
    for i in range(len(sponge_stats.loss)):
        sponge_loss += sponge_stats.loss[i].to('cuda')
        fired += float(sponge_stats.fired[i])
        fired_perc += float(sponge_stats.fired_perc[i])
        l2 += float(sponge_stats.l2[i])
    remove_hooks(hooks)

    sponge_loss /= len(sponge_stats.loss)
    fired_perc /= len(sponge_stats.loss)

    sponge_loss *= args.sponge_lb
    return sponge_loss, outputs, (float(sponge_loss), fired, fired_perc, l2)


def compute_sponge_loss(model, inputs, victim_leaf_nodes,args):
    sponge_loss, _, sponge_stats = do_sponge_loss(model, inputs, victim_leaf_nodes,args)
    sponge_stats = dict(sponge_loss=float(sponge_loss), sponge_stats=sponge_stats)
    return sponge_loss, sponge_stats