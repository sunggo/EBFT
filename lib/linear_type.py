import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .options import args
N = args.N
M = args.M
# class GetMask(autograd.Function):
#     @staticmethod
#     def forward(ctx, mask, prune_rate):
#         out = mask.clone()
#         _, idx = mask.abs().flatten().sort()
#         j = int(prune_rate * mask.numel())

#         # flat_out and out access the same memory.
#         flat_out = out.flatten()
#         flat_out[idx[:j]] = 0
#         flat_out[idx[j:]] = 1
#         return out

#     @staticmethod
#     def backward(ctx, g):
#         # send the gradient g straight-through on the backward pass.
#         return g, None
# class GetMaskNM(autograd.Function):
#     @staticmethod
#     def forward(ctx, mask):
#         out = mask.clone()
#         length = mask.numel()
#         group = int(length / M)
#         out_regroup = out.view(group, M)
#         indices = torch.argsort(out_regroup.abs(), dim=1,descending=True)[:, :N]
#         out_regroup = out_regroup.zero_().scatter_(1, indices.cuda(), 1)
#         return out

#     @staticmethod
#     def backward(ctx, g):
#         # send the gradient g straight-through on the backward pass.
#         return g, None

class LinearMasked(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.zeros(self.weight.shape))

    def forward(self, x):
        if self.prune_rate != 0:
            # if N != 0 :
            #     mask = GetMaskNM.apply(self.mask)
            # else:
            #     mask = GetMask.apply(self.mask, self.prune_rate)
            sparseWeight = self.mask * self.weight
            x = F.linear(
                x, sparseWeight, self.bias
            )
        else:
            x = F.linear(
                x, self.weight, self.bias
            )
        return x
