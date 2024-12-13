# import torch 



# class ContractBGSpaceAutoGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         """
#         x ... x 3
#         """
#         abs_x = torch.abs(x)
#         L_infinity_norm, index = torch.max(abs_x, dim=-1, keepdim=True)
#         if (L_infinity_norm == 0).sum() > 0:
#             invalid = L_infinity_norm[...,0] == 0
#             print(x[invalid])
#             raise ValueError
#         mask = torch.zeros(abs_x.shape, dtype=torch.bool, device=x.device)
#         mask[torch.arange(mask.shape[0]), index.flatten()] = True
#         return torch.where(mask, (2. - 1./abs_x)*torch.sign(x), x / L_infinity_norm)

#     @staticmethod
#     def backward(ctx, grad_in):
#         """
#         ... x 3
#         """
#         return grad_in

# def ContractBGSpace(x):
#     return ContractBGSpaceAutoGrad.apply(x)
