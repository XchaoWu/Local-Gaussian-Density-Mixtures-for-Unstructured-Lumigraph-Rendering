import torch 
import torch.nn as nn 
import math 
from .lib.HASHGRID import (
    encoding_forward_cuda,
    encoding_backward_cuda
) 

class HashEncodingAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, hash_idxs, features, resolution):
        
        batch_size = points.shape[0]
        n_levels = features.shape[1]

        outputs = torch.full((batch_size, n_levels, 2), 0, dtype=torch.float32, device=points.device)
        encoding_forward_cuda(points, hash_idxs, outputs, features, resolution)
    


        ctx.save_for_backward(points, hash_idxs, features, resolution)

        return outputs
    
    @staticmethod
    def backward(ctx, grad_in):
        points, hash_idxs, features, resolution = ctx.saved_tensors
        grad_features = torch.zeros_like(features)
        encoding_backward_cuda(points, hash_idxs, grad_in, grad_features, features, resolution)
        return None, None, grad_features, None
    
def HashEncoding(points, hash_idxs, features, resolution):
        
    return HashEncodingAutoGrad.apply(points, hash_idxs, features, resolution)

    
class HashGrid2D(nn.Module):
    def __init__(self, num_hashgrid=1, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=(16,16), finest_resolution=(512,512),
                 init_mode="xavier"):
        super(HashGrid2D, self).__init__()

        assert n_features_per_level == 2, "we only support dim=2"

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.out_dim = self.n_levels * self.n_features_per_level
        self.num_hashgrid = num_hashgrid

        self.b_w = math.exp((math.log(self.finest_resolution[0])-math.log(self.base_resolution[0]))/(n_levels-1))
        self.b_h = math.exp((math.log(self.finest_resolution[1])-math.log(self.base_resolution[1]))/(n_levels-1))
        # self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))


        resolution_w = []
        resolution_h = []
        for i in range(self.n_levels):
            resolution_w += [int(self.base_resolution[0] * self.b_w**i)]
            resolution_h += [int(self.base_resolution[1] * self.b_h**i)]
        
        resolution_w = torch.tensor(resolution_w, dtype=torch.int32)
        resolution_h = torch.tensor(resolution_h, dtype=torch.int32)
        resolution = torch.stack([resolution_w,resolution_h], dim=-1)

        
        self.register_buffer("resolution", resolution)
        
        self.features = torch.zeros(self.num_hashgrid, self.n_levels, 2**self.log2_hashmap_size, self.n_features_per_level,
                                    dtype=torch.float32)


        self.features = nn.Parameter(self.features)

        if init_mode == 'kaiming':
            nn.init.kaiming_normal_(self.features)
        elif init_mode == 'xavier':
            nn.init.xavier_normal_(self.features)
        elif init_mode == 'uniform':
            nn.init.uniform_(self.features, -1e-4, 1e-4)
        print(f"{init_mode} init feature")
        

    def forward(self, x, hash_idxs):
        """
        x ... x 2
        hash_idxs ... x 1
        return ... x 32 
        """
        ori_shape = x.shape[:-1]
        
        features = HashEncoding(x.reshape(-1,2), hash_idxs.reshape(-1,1), self.features, self.resolution)
        features = features.reshape(*list(ori_shape), self.n_levels*self.n_features_per_level)
        return features


    





if __name__ == "__main__":
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")

    phg = HashGrid2D(1).to(device)

    # x = torch.randn(1,2, device=device, dtype=torch.float32)
    x = torch.full((1,2), -0.1, device=device, dtype=torch.float32)
    hash_idxs = torch.tensor([0], device=device, dtype=torch.int32)
    x.requires_grad_(True)
    import time 
    s = time.time()
    f = phg(x, hash_idxs)
    print(f)
    # torch.cuda.synchronize()
    # e = time.time()
    # print(e - s)
    # print(f.shape)

    # f.sum().backward()
    # print(x.grad)
    
