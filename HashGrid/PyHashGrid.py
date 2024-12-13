import torch 
import torch.nn as nn 
import math 
from .lib.HASHGRID import (
    embedding_forward_cuda,
    embedding_backward_cuda
) 

class HashEmbeddingAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, features, block_corner, block_size, resolution):
        
        batch_size = points.shape[0]
        n_levels = features.shape[0]

        outputs = torch.full((batch_size, n_levels, 2), 0, dtype=torch.float32, device=points.device)
        embedding_forward_cuda(points, outputs, features, block_corner, block_size, resolution)

        ctx.block_size = block_size
        ctx.save_for_backward(points, features, block_corner, resolution)

        return outputs
    
    @staticmethod
    def backward(ctx, grad_in):
        points, features, block_corner, resolution = ctx.saved_tensors
        grad_points = torch.zeros_like(points)
        grad_features = torch.zeros_like(features)
        embedding_backward_cuda(points, grad_in, grad_points, grad_features, features, block_corner, ctx.block_size, resolution)
        return grad_points, grad_features, None, None, None 
    
def HashEmbedding(points, features, block_corner, block_size, resolution):
    return HashEmbeddingAutoGrad.apply(points, features, block_corner, block_size, resolution)
            
    
class PyHashGrid(nn.Module):
    def __init__(self, bbox_corner, bbox_size, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512,
                 init_mode="xavier"):
        super(PyHashGrid, self).__init__()

        assert n_features_per_level == 2, "we only support dim=2"

        self.bbox_corner = bbox_corner
        self.bbox_size = bbox_size

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.out_dim = self.n_levels * self.n_features_per_level

        # self.b = math.exp((math.log(self.finest_resolution)-math.log(self.base_resolution))/(n_levels-1))
        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))


        resolution = []
        for i in range(self.n_levels):
            resolution += [(self.base_resolution * self.b**i).int()]
        resolution = torch.stack(resolution, 0)
        self.register_buffer("resolution", resolution)
        
        self.features = torch.zeros(self.n_levels, 2**self.log2_hashmap_size, self.n_features_per_level,
                                    dtype=torch.float32)
        self.features = nn.Parameter(self.features)

        if init_mode == 'kaiming':
            nn.init.kaiming_normal_(self.features)
        elif init_mode == 'xavier':
            nn.init.xavier_normal_(self.features)
        elif init_mode == 'uniform':
            nn.init.uniform_(self.features, -1e-4, 1e-4)
        print(f"{init_mode} init feature")

    def forward(self, x):
        """
        x ... x 3
        return ... x 32 
        """
        ori_shape = x.shape[:-1]
        features = HashEmbedding(x.reshape(-1,3), self.features, self.bbox_corner, self.bbox_size, self.resolution)
        features = features.reshape(*list(ori_shape), self.n_levels*self.n_features_per_level)
        return features


    





if __name__ == "__main__":
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")
    box_corner = torch.tensor([0,0,0], dtype=torch.float32).to(device)
    bbox_size = 1
    phg = PyHashGrid(device, box_corner, bbox_size)

    x = torch.randn(1,3, device=device, dtype=torch.float32)
    x.requires_grad_(True)
    import time 
    s = time.time()
    f = phg(x)
    torch.cuda.synchronize()
    e = time.time()
    print(e - s)
    print(f.shape)

    f.sum().backward()
    print(x.grad)
    
