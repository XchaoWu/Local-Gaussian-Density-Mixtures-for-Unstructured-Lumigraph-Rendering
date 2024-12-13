import torch 
import numpy as np 
from tools import utils 


class RaySample:
    def __init__(self, num_view, H, W, device):
        self.progress_img_ref = np.zeros((num_view, H, W, 3), dtype=np.float32)
        self.progress_img = np.zeros((num_view, H, W, 3), dtype=np.float32)
        self.error_map_ref = torch.ones((num_view, H, W, 1), dtype=torch.float32, device=device)
        self.error_map = torch.ones((num_view, H, W, 1), dtype=torch.float32, device=device)
        
        self.num_view = num_view
        self.H = H 
        self.W = W 
        self.device = device
    
    def update(self, b_idx, ray_idx, pred_color_ref, pred_color, gt_color):
        H = self.H 
        W = self.W
        self.progress_img_ref.reshape(-1, H*W, 3)[b_idx.cpu().numpy(), ray_idx.cpu().numpy(), :] = \
            pred_color_ref.detach().cpu().reshape(-1, len(ray_idx),3) * 255

        self.progress_img.reshape(-1, H*W, 3)[b_idx.cpu().numpy(), ray_idx.cpu().numpy(), :] = \
            pred_color.detach().cpu().reshape(-1, len(ray_idx),3) * 255
        
        self.error_map_ref.reshape(-1, H*W, 1)[b_idx, ray_idx, :] = \
            torch.abs(pred_color_ref.detach() - gt_color).mean(dim=-1,keepdim=True).reshape(-1, len(ray_idx),1)
        
        self.error_map.reshape(-1, H*W, 1)[b_idx, ray_idx, :] = \
            torch.abs(pred_color.detach() - gt_color).mean(dim=-1,keepdim=True).reshape(-1, len(ray_idx),1)
        # temp_error = torch.abs(pred_color.detach() - gt_color).mean(dim=-1,keepdim=True).reshape(-1, len(ray_idx),1)
        # self.error_map.reshape(-1, H*W, 1)[b_idx, ray_idx, :] = 1. - torch.exp(-5. * temp_error)    
    

    def sample_rays_with_distribution(self, H, W, probs, sample_num, num_view):
        
        if probs != None:
            probs = probs.reshape(probs.shape[0],-1)
            idxs = probs.multinomial(num_samples=sample_num)
        else:
            idxs = []
            for i in range(num_view):
                idxs += [torch.randperm(H*W, device=self.device)[:sample_num]]
            idxs = torch.stack(idxs)
            
        b_idxs = torch.arange(idxs.shape[0], device=self.device)[:, None].repeat(1, idxs.shape[1])
        
        return b_idxs.flatten(), idxs.flatten()

    def random_sampling(self, sample_num):

        ray_idx = torch.randperm(self.H*self.W, device=self.device)[:sample_num]
        # num_view x sample_num
        ray_idx = ray_idx[None,:].repeat(self.num_view, 1)
        return ray_idx
    
    def important_sampling(self, sample_num):
        probs = self.error_map.reshape(self.num_view,-1)
        # num_view x sample_num
        ray_idx = probs.multinomial(num_samples=sample_num)
        
        return ray_idx
    
    def sample_rays(self, batch_size, num_neighbor):

        sample_num = batch_size // self.num_view
        
        important_ratio = 0
        
        important_num = int(sample_num * important_ratio)
        random_num = sample_num - important_num
        
        
        ray_idx = []
        if important_num > 0:
            import_ray_idx = self.important_sampling(important_num)
            ray_idx += [import_ray_idx]
        
        if random_num > 0:
            random_ray_idx = self.random_sampling(random_num)
            ray_idx += [random_ray_idx]
        
        ray_idx = torch.cat(ray_idx, -1).flatten()
        
        # ray_idx = torch.randperm(self.H*self.W, device=self.device)[:sample_num]
        # ray_idx = ray_idx[None,:].repeat(self.num_view, 1).flatten()
        b_idx = torch.arange(self.num_view, device=self.device)[:, None].repeat(1, sample_num).flatten()
        return b_idx, ray_idx 