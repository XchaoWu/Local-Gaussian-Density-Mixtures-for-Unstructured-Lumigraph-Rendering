import torch 
import torch.nn as nn  
import numpy as np 
import network 
from cuda import grid_sample_forward_cuda, grid_sample_backward_cuda, gaussian_grid_sample_forward_cuda, gaussian_grid_sample_backward_cuda
from cuda import computeViewcost, warping_samples_cuda, neighbor_score_cuda, pick_up_neighbor, new_neighbor_score_cuda, gaussian_sample, grid_sample_feature
from cuda import get_candidate_neighbor, pixel_level_neighbor_ranking, pixel_level_pick_up_neighbor
from cuda import get_warping_mask_cuda, get_candidate_uniform_neighbor
from cfg import * 
import torch.nn.functional as F 
from torchvision import transforms
import random 
import camera 
import cv2, os 
import time 
from HashGrid import HashGrid,PyHashGridBG
import matplotlib.pyplot as plt 
from tools import tools, utils
from tqdm import tqdm 
from occupied_grid import OccupiedGrid
from cuda import ray_aabb_intersection, padding_depth
from easydict import EasyDict as edict
from HashGrid import PyHashGrid
from tools.utils import TIME_TEST

POINT_VIEW_SELECTION = 0
RAY_VIEW_SELECTION = 1
VIEW_VIEW_SELECTION = 2

"""
Geometry should be consistent in global 
Blending should use a small number of images 
"""

class LGDM(nn.Module):
    def __init__(self, num_view, images, num_neighbor, num_gaussian, 
                 data_type,
                 hierarchical_sampling=False,
                 device=None, **kwargs):
        super(LGDM, self).__init__()

        self.device = device 
        self.num_view = num_view
        
        
        self.hierarchical_sampling = hierarchical_sampling
        
        
        self.deg = 3
        
        self.use_ref = False
        
        self.data_type = data_type
        
        # if init_near_far != None:
        #     self.init_near_far = init_near_far.to(self.device).permute(0,3,1,2)
        # else:
        #     self.init_near_far = None     

        # self.blendingNet = network.BlendingNet(num_blend=num_neighbor).to(self.device)
        # network.init_model(self.blendingNet, "xavier")

        self.coeffiNet = network.ColorNet(num_neighbor, self.deg).to(self.device)
        network.init_model(self.coeffiNet, "xavier")

        # self.specularNet = network.SpecularNet().to(self.device)
        # network.init_model(self.specularNet, "xavier")
        
        self.images = torch.from_numpy(images * 255).type(torch.uint8).to(self.device)


        self.H, self.W = self.images.shape[1:3]
        
        self.padding = 50
        
        
        self.valid_mask = torch.ones((self.num_view, self.H, self.W, 1), dtype=torch.bool, device=self.device)
        
        self.per_view_depth = torch.ones((self.num_view, self.H, self.W, 1), dtype=torch.float32, device=self.device) * 1e8
        
        # self.ogrid = OccupiedGrid(8, device)
        
        self.num_gaussian = num_gaussian
        
        # if self.data_type == SCENE360:
        #     self.num_fg_gaussian = 10
        #     self.num_bg_gaussian = self.num_gaussian - self.num_fg_gaussian
        
        # assert self.num_bg_gaussian > 2

        self.fine_model = network.RayGaussian(num_view=self.num_view,num_gaussian=self.num_gaussian,
                                        height=self.H, width=self.W, padding=self.padding).to(self.device)
        network.init_model(self.fine_model, "xavier")
        # self.proposal_model = network.ProposalGaussian(num_view=self.num_view,num_gaussian=self.num_gaussian,
        #                                 height=self.H, width=self.W).to(self.device)
        # network.init_model(self.proposal_model, "xavier")
        
        self.model = self.fine_model
        

        # self.filter_scale = 1.0

        
        
        # if self.hierarchical_sampling:
        #     self.proposal_net = network.ProposalNetwork().to(self.device)
        #     network.init_model(self.proposal_net, "xavier")
        
        # self.gaussian_mask = torch.ones((self.num_view, self.H, self.W, self.num_gaussian), dtype=torch.bool, device=self.device)
        
        
        # self.blendingNet = network.BlendingNet(num_blend=num_neighbor, 
        #                                        activation=nn.ReLU()).to(self.device)

        self.num_neighbor = num_neighbor
        
        # self.num_candidate = min(self.num_neighbor * 3, self.num_view)
        # self.num_candidate = min(self.num_neighbor * 10, self.num_view)
        
        # self.num_candidate = min(self.num_neighbor * 3, self.num_view)
        self.num_candidate = self.num_view
        
        # self.num_candidate = self.num_view
        
        self.weight_gaussian = None 
        
        # self.beta = nn.Parameter(torch.full((1,), 1.0, dtype=torch.float32, device=self.device))
        # self.gaussian_inv_sigma = 1.0
        # self.beta = nn.Parameter(torch.full((1,), 1.0, dtype=torch.float32, device=self.device))

        self.blur_para = (10.0, 10.0)

        self.ogrid = None 
        
        self.gausssian_prunning_weight = 0.

    @torch.no_grad()
    def get_valid_mask(self, camera):
        
        return 
        
        rts = camera.get_rts()
        ks = camera.ks
        bs = 2 ** 14
        num_sample = 192
        print("getting valid mask")
        for k in tqdm(range(camera.num_camera)):
            rays_o, rays_d, _ = camera.get_rays(view_idx=k)
            rays_o = rays_o.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)
            ref_idxs = torch.ones_like(rays_o[...,:1]) * k 
            ref_idxs = ref_idxs.int()
            z_vals, _, _ = self.ogrid.near_far_grid_sampling(rays_o, rays_d, num_sample)
            samples = rays_o[:,None,:] + z_vals[...,None] * rays_d[:,None,:]
            candidate_neighbors = self.get_candidate_neighbor(rays_o, camera, ref_idxs)
               
            for i in range(0, rays_o.shape[0], bs):
                mask = torch.full((rays_o[i:i+bs].shape[0], num_sample, self.num_candidate, 1), 0, dtype=torch.bool, device=self.device)
                get_warping_mask_cuda(ref_idxs[i:i+bs], rays_o[i:i+bs], rays_d[i:i+bs],
                                      samples[i:i+bs], rts, ks, candidate_neighbors[i:i+bs], 
                                      mask, self.H, self.W)
                valid_mask = (mask.sum(2) > 0).sum(1) > (num_sample / 2)
                self.valid_mask[k].reshape(-1,1)[i:i+bs] = valid_mask
        torch.cuda.empty_cache()
    
    
    
    def create_per_view_uv(self):
        y_range = torch.arange(self.H,dtype=torch.float32,device=self.device)
        x_range = torch.arange(self.W,dtype=torch.float32,device=self.device)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        uv = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
        # uv[..., 0] = uv[..., 0] / (self.W - 1) # [0,1]
        # uv[..., 1] = uv[..., 1] / (self.H - 1) # [0,1]

        # uv[..., 0] = (uv[..., 0] + self.padding + 0.5) / (self.W + 2 * self.padding) # [0,1]
        # uv[..., 1] = (uv[..., 1] + self.padding + 0.5) / (self.H + 2 * self.padding) # [0,1]

        uv[..., 0] = (uv[..., 0] + self.padding) / (self.W + 2 * self.padding) # [0,1]
        uv[..., 1] = (uv[..., 1] + self.padding) / (self.H + 2 * self.padding) # [0,1]
                
        return uv 
    
        
    def export(self, save_dir):
        
        
        """
        
        perview: 
        离散化的 gaussian mixture   
        假设有 K 个 gaussian 
        Gaussian Para
        N x H x W * (3*K)
        Coeffi Net 网络
        Occupied Grid 
        
        
        """
        
        # 1. occupied grid
        grid_mask = self.ogrid.grid_mask.detach().cpu().numpy()
        # np.save(os.path.join(save_dir, f"occupied_grid.npy"), grid_mask)
        np.savez(os.path.join(save_dir, f"occupied_grid.npz"), 
                 occupied_grid=grid_mask,
                 bbox_corner=self.ogrid.bbox_corner.detach().cpu().numpy(), 
                 bbox_size=self.ogrid.bbox_size.detach().cpu().numpy())
        
        
        # coeffi Net 
        torch.save(self.coeffiNet.state_dict(), os.path.join(save_dir, f"coeffiNet.pt"))
        
        
        uv = self.create_per_view_uv()
        
        for idx in range(self.num_view):
            # B x 1 
            netIdx = torch.ones_like(uv[..., :1]) * idx
            rgb = self.images[idx].float().reshape(-1,3) / 255.
            mu_texture = torch.zeros((self.H*self.W, self.num_gaussian), dtype=torch.float32, device=self.device)
            inv_sigma_texture = torch.zeros((self.H*self.W, self.num_gaussian), dtype=torch.float32, device=self.device)
            weight_texture = torch.zeros((self.H*self.W, self.num_gaussian), dtype=torch.float32, device=self.device)
            
            bs = 2**14 
            for i in range(0, uv.shape[0], bs):
                mu, inv_sigma, weight, _, _ = self.infer_net(uv[i:i+bs], netIdx[i:i+bs], None)
                mu_texture[i:i+bs] = mu 
                inv_sigma_texture[i:i+bs] = inv_sigma
                weight_texture[i:i+bs] = weight
            
            mu_texture = mu_texture.detach().cpu().numpy().reshape(self.H, self.W, self.num_gaussian)
            inv_sigma_texture = inv_sigma_texture.detach().cpu().numpy().reshape(self.H, self.W, self.num_gaussian)
            weight_texture = weight_texture.detach().cpu().numpy().reshape(self.H, self.W, self.num_gaussian)

            path = os.path.join(save_dir, f"{idx}.npz")
            np.savez(path, mu_texture=mu_texture, inv_sigma_texture=inv_sigma_texture, 
                     weight_texture=weight_texture)
            cv2.imwrite(os.path.join(save_dir, f"{idx}.png"), self.images[idx].detach().cpu().numpy()[...,::-1])
            print(f"save {path} successfully\n")

        
        
        

    def set_ogrid(self, ogrid):
        self.ogrid = ogrid
        

    
    def s_space_to_t(self, s):
        invert = lambda x: 1.0 / x
        return invert(invert(self.ogrid.near) * (1-s) + invert(self.ogrid.far) * s)
    
    def t_sapce_to_s(self, t):
        invert = lambda x: 1.0 / x
        return (invert(t) - invert(self.ogrid.near)) / (invert(self.ogrid.far) - invert(self.ogrid.near))
    
    def get_dists(self, rays_d, z_vals):
        dists = torch.cat([z_vals[...,1:] - z_vals[...,:-1], 1e10*torch.ones(rays_d.shape[0], 1, device=self.device)], dim=-1)
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        return dists
    
    def ray_contraction(self, z_vals):
        near = self.ogrid.near 
        far = self.ogrid.far 
        sqrt3 = 3 ** 0.5
        return torch.where(z_vals <= sqrt3, (z_vals - near) / sqrt3, 2 - (sqrt3 + near) / z_vals)
    
    def inv_ray_contraction(self, z_vals):
        near = self.ogrid.near 
        far = self.ogrid.far 
        sqrt3 = 3 ** 0.5
        return torch.where(z_vals <= 1, z_vals * sqrt3 + near, (sqrt3 + near) / (2 - z_vals))
    
    @torch.no_grad()
    def sample_points(self, rays_o, rays_d, num_sample, mode):

        s = torch.linspace(0., 1., steps=num_sample, device=self.device)   
        
        t = self.s_space_to_t(s)
        
        z_vals = t[None,:].repeat(rays_o.shape[0], 1)
        

            
        
        dists = self.get_dists(rays_d, z_vals)
        
        return z_vals, dists
    
        if self.data_type == LLFF:
            raise NotImplementedError
            # return self.ogrid.sample_points_LLFF(rays_o, rays_d, num_sample)
        elif self.data_type == SCENE360:
            # [FIXME] here\
            return self.ogrid.sample_points_360(rays_o, rays_d, num_sample)
            # return self.ogrid.sample_points_LLFF_v2(rays_o, rays_d, num_sample)
            # return self.ogrid.sample_points_360(rays_o, rays_d, num_sample)



    
    @torch.no_grad()
    def get_candidate_neighbor(self, rays_o, cam, ref_idxs):
        batch_size = rays_o.shape[0]
        distance = torch.full((batch_size, self.num_view), 1e10, dtype=torch.float32, device=self.device)
        get_candidate_neighbor(ref_idxs, rays_o, cam.get_rts(), cam.ks, distance)
        
        sorted_distance, sorted_idxs = torch.sort(distance, dim=-1, descending=False)
        
        # B x num_candidate 
        candidate_neighbors = sorted_idxs[:,:self.num_candidate]
        
        # candidate_neighbors = torch.arange(0, self.num_candidate, device=self.device)[None, :].repeat(batch_size,1)
        # print(candidate_neighbors[0])
        # input()
        return candidate_neighbors.int()
    
    # @TIME_TEST
    @torch.no_grad()
    def neighbor_view_selection_v3(self, rays_o, rays_d, z_vals, cam, mode, up_axis, ref_idxs):

        if mode is INFERENCE:
            ref_idxs = torch.ones_like(rays_o[...,:1]).int() * -1
            
        B, N = z_vals.shape[:2]
        num_neighbor = self.num_neighbor
        
        # # s = time.time()
        candidate_neighbors = self.get_candidate_neighbor(rays_o, cam, ref_idxs)

        if mode == INFERENCE:
            padding = 0
        else:
            padding = self.padding

        step = N
        num_step = N//step
        score = torch.full((B, num_step, self.num_candidate, 4), 
                           0.0, dtype=torch.float32, device=self.device)
        pixel_level_neighbor_ranking(ref_idxs, candidate_neighbors, cam.get_rts(), cam.ks, 
                                     rays_o, rays_d, z_vals, up_axis, score, step, padding, self.H, self.W)
        
    
        
        # B x num_step x num_candidate x 4
        sorted_score, sorted_idxs = torch.sort(score, dim=2, descending=True)
        sorted_idxs[sorted_score <= 0] = -1
        sorted_idxs = sorted_idxs.int()

        nei_idxs = torch.full((B, num_step, self.num_candidate, 1), -1, dtype=torch.int32, device=self.device)
        # s = time.time()
        pick_up_neighbor(sorted_idxs.int(), candidate_neighbors, sorted_score, nei_idxs)
        
        # nei_idxs = torch.full((B, self.num_candidate, 1), -1, dtype=torch.int32, device=self.device)
        # pixel_level_pick_up_neighbor(sorted_idxs, candidate_neighbors, nei_idxs)
        
        if mode is TRAIN:
            temp = torch.rand(*nei_idxs.shape, dtype=torch.float32, device=self.device)
            num_candidate = min(num_neighbor * random.randint(1,3), self.num_candidate-1)
            temp[nei_idxs==-1] = -1.0 # 不可见图片设置-1, 其他值位于[0,1)
            temp[:,:, num_candidate:,:] = -1.0
            # B x num_sample x num_neighbor x 1 
            idxs = torch.argsort(temp, dim=2, descending=True)[:,:, :num_neighbor]
            mask = torch.zeros_like(nei_idxs).scatter_(dim=2, index=idxs, src=torch.ones_like(nei_idxs)).bool()
            nei_idxs = nei_idxs[mask].reshape(*idxs.shape)
        else:
            # B x num_sample x num_neighbor x 1
            nei_idxs = nei_idxs[:,:,:num_neighbor]

        
        # for item in nei_idxs.detach().cpu().flatten().numpy():
        #     print(item)
        #     if item == -1:
        #         continue 
        #     img = self.images[item].detach().cpu().numpy()
        #     cv2.imwrite(f"{item}.png", img)

        # for item in candidate_neighbors.detach().cpu().flatten().numpy():
        #     print(item)
        #     if item == -1:
        #         continue 
        #     img = self.images[item].detach().cpu().numpy()
        #     cv2.imwrite(f"{item}.png", img)
            
        # exit()
                
        # print(nei_idxs[0])
        nei_idxs = nei_idxs.repeat_interleave(step, dim=1)
        # print(nei_idxs[0])
        # print(nei_idxs.shape)
        # exit()

        return nei_idxs

        
        
    @torch.no_grad()
    def neighbor_view_selection_v2(self, rays_o, rays_d, samples, cam, mode, up_axis, ref_idxs):
        
        INVALID_SCORE = -1e10
        
        if mode is INFERENCE:
            ref_idxs = torch.ones_like(rays_o[...,:1]).int() * -1
            
        B, N = samples.shape[:2]
        num_neighbor = self.num_neighbor
        
        # # s = time.time()
        candidate_neighbors = self.get_candidate_neighbor(rays_o, cam, ref_idxs)

        # # torch.cuda.synchronize()
        # # e = time.time()
        # # t = (e - s) * 1000
        # # print(f"{t}ms")

        # batch_size x num_sample x num_candidate x 1 
        temp_score = torch.full((B, N, self.num_candidate, 4), 0, dtype=torch.float32, device=self.device)
        
        # s = time.time()
        
        if mode==INFERENCE:
            padding = 0
        else:
            padding = self.padding
        new_neighbor_score_cuda(ref_idxs, up_axis, rays_o, rays_d, samples,
                                cam.get_rts(), cam.ks, candidate_neighbors, temp_score,
                                padding, self.H, self.W)

        # B x num_sample x num_candidate x 4
        # s = time.time()
        temp_score, temp_nei_idxs = torch.sort(temp_score, dim=2, descending=True)
        
        
        # if mode is TRAIN:
        #     temp_score[temp_score <= 0] = INVALID_SCORE
        temp_nei_idxs[temp_score <= 0] = -1
        # torch.cuda.synchronize()
        # e = time.time()
        # t = (e - s) * 1000
        # print(f"{t}ms")
                    
        # B x num_sample x num_candidate x 1
        nei_idxs = torch.full((B, N, self.num_candidate, 1), -1, dtype=torch.int32, device=self.device)
        # s = time.time()
        pick_up_neighbor(temp_nei_idxs.int(), candidate_neighbors, temp_score, nei_idxs)

        del temp_score, temp_nei_idxs
            
        if mode is TRAIN:
            temp = torch.rand(*nei_idxs.shape, dtype=torch.float32, device=self.device)
            num_candidate = min(num_neighbor * random.randint(1,3), self.num_candidate-1)
            temp[nei_idxs==-1] = -1.0 # 不可见图片设置-1, 其他值位于[0,1)
            temp[:,:, num_candidate:,:] = -1.0
            # B x num_sample x num_neighbor x 1 
            idxs = torch.argsort(temp, dim=2, descending=True)[:,:, :num_neighbor]
            mask = torch.zeros_like(nei_idxs).scatter_(dim=2, index=idxs, src=torch.ones_like(nei_idxs)).bool()
            nei_idxs = nei_idxs[mask].reshape(*idxs.shape)
        else:
            # B x num_sample x num_neighbor x 1
            nei_idxs = nei_idxs[:,:,:num_neighbor]

        
        return nei_idxs
    
    

    @torch.no_grad()
    def draw_ray_curve(self, cam, idx, num_sample, loc):
        raise NotImplementedError
        
        uv = torch.tensor([loc[0],loc[1]], dtype=torch.float32, device=self.device)
        uv = uv.reshape(-1,2)
        
        # B x 1 
        netIdx = torch.ones_like(uv[..., :1]) * idx
        
        ridx = (uv[..., 1] * self.W + uv[..., 0]).long()
        rays_o, rays_d, _ = cam.get_rays(view_idx=idx)
        rays_o = rays_o.reshape(-1,3)[ridx]
        rays_d = rays_d.reshape(-1,3)[ridx]
        
        uv[..., 0] = (uv[..., 0]+0.5) / self.W
        uv[..., 1] = (uv[..., 1]+0.5) / self.H
        
        # B x 2
        # bounds = torch.full((rays_o.shape[0], 2), -1, dtype=torch.float32, device=self.device)
        # ray_aabb_intersection(rays_o, rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8), 
        #                       self.ogrid.bbox_center, self.ogrid.bbox_size, bounds)
        # B x num_sample 
        z_vals, dists, bounds = self.sample_points(rays_o, rays_d, num_sample)
        
        # print("这里需要DEBUG")
        # B x num_sample x 1 [FIXME]
        ray_distance = z_vals * torch.norm(rays_d, dim=-1, keepdim=True)
        ray_distance = ray_distance[..., None]

        alpha, visibility, _,_,_,_,_  = \
            self.infer_(uv, netIdx, None,
                        ray_distance, bounds, INFERENCE)
        
        # alpha = 1. - torch.exp(-alpha * dists[..., None])
        # weight = self.cal_weight(alpha)
    
        weight = self.cal_ref_weight(alpha, visibility)
        
        
        print("visibility", visibility.detach().cpu().numpy())
        
        depth = torch.sum(weight * z_vals[..., None])
        print("depth", depth)
        
        # print(alpha.flatten().detach().cpu().numpy())
        import matplotlib.pyplot as plt 
        plt.plot(z_vals.flatten().detach().cpu().numpy(), alpha.flatten().detach().cpu().numpy(), color="green")
        plt.plot(z_vals.flatten().detach().cpu().numpy(), visibility.flatten().detach().cpu().numpy(), color="blue")
        plt.grid()
        plt.legend()
        plt.show()
        
    @torch.no_grad()
    def update_rendered_per_view_depth(self, cam, num_sample):
        return 
        # self.ogrid.clear()
        for idx in tqdm(range(self.num_view), desc="updating per view depth"):
            tdepth, surface_points = \
                self.render_per_view_depth(cam, idx, num_sample)
            self.per_view_depth[idx] = tdepth.reshape(self.H, self.W, 1)
            # self.ogrid.update_value(surface_points, torch.ones_like(surface_points[...,:1]),gamma=0.0)
            
        # pad_rendered_depth = self.per_view_depth.clone()
        # padding_depth(self.per_view_depth, self.valid_mask, pad_rendered_depth, 50)
        # self.per_view_depth = pad_rendered_depth
    
    @torch.no_grad()
    def render_per_view_depth(self, cam, idx, num_sample):
        
        raise NotImplementedError
    
        uv = self.create_per_view_uv()
        # B x 1 
        netIdx = torch.ones_like(uv[..., :1]) * idx
        
        rays_o, rays_d, _ = cam.get_rays(view_idx=idx)
        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)
        
        rgb = self.images[idx].float().reshape(-1,3) / 255.
        
        depth = torch.zeros_like(uv[..., :1])
        # mu = torch.zeros_like(uv[..., :1])
        
        bs = 2**14 
        for i in range(0, rays_o.shape[0], bs): 
            # B x num_sample 
            z_vals, dists, bounds = self.sample_points(rays_o[i:i+bs], rays_d[i:i+bs], num_sample)
            
            # B x num_sample x 1
            ray_distance = z_vals * torch.norm(rays_d[i:i+bs], dim=-1, keepdim=True)
            ray_distance = ray_distance[..., None]
            alpha, visibility, _,_,_,_,_  = \
                self.infer_(uv[i:i+bs,None,:], netIdx[i:i+bs,None,:], rgb[i:i+bs,None,:],
                            ray_distance, bounds[:,None,:], INFERENCE)

            weight = self.cal_ref_weight(alpha, visibility)
            dep = torch.sum(weight * z_vals[..., None], dim=1)
            
            depth[i:i+bs] = dep 
        
        surface_points = rays_o + depth * rays_d
        
        return depth, surface_points
        
    
    def infer_net(self, uv, netIdx, nei_dir):
        
        
        out = self.model(uv, netIdx.int(), nei_dir)
        mu = out["mu"]
        inv_sigma = out["inv_sigma"]
        weight = out["weight"] 
        feature = out["feature"]
        warped_pred_color = out["warped_pred_color"]

        
        return mu, inv_sigma, weight, feature, warped_pred_color
    
    def infer_(self, uv, netIdx, nei_dir, z, nei_bound, mode=TRAIN):
        
        """
        N x C x H x W
        N x B x 1 x 2
        """    
        norm_mu, inv_sigma, weight, feature, warped_pred_color = self.infer_net(uv, netIdx, nei_dir)
    

        near = nei_bound[..., 0:1]
        far  = nei_bound[..., 1:2]
        
            
        mu = 1.0 / (1.0 / (near+1e-8) * (1.0 - norm_mu) + 1.0 / (far+1e-8) * norm_mu)
            
        
        z_dists = z - mu 
        
        
        alpha = torch.sum(weight * self.gaussian_func(z, mu, inv_sigma), dim=-1, keepdim=True)
        visibility = torch.exp(-torch.sum(weight * self.intergrated_gaussian_func(z, mu, inv_sigma, near), dim=-1, keepdim=True))
        
        return alpha, visibility, norm_mu, inv_sigma, z_dists, feature, weight, warped_pred_color



    def gaussian_func(self, x, mu, inv_sigma):
        pi = 3.1415926 
        return inv_sigma / (((2*pi) ** 0.5)) * torch.exp(-0.5 * (((x - mu) * inv_sigma) ** 2) )

    def intergrated_gaussian_func(self, x, mu, inv_sigma, near):
        func = lambda t: 0.5 * torch.erf(inv_sigma * (t - mu) / (2 ** 0.5)) + 0.5
        # B x num_sample x num_neighbor x 1 
        # return func(x)

        res = func(x) - func(near)
        res[res < 0] = 0
        return res

    
    def density_function(self, x, mu, weight):
        pi = 3.1415926 
        gaussian_func = lambda inv_sigma,mu,t: inv_sigma / (((2*pi) ** 0.5)) * torch.exp(-0.5 * (((t - mu) * inv_sigma) ** 2) )
        
        temp = []
        for i in range(weight.shape[-1]):
            temp += [gaussian_func(2**i, mu[...,i:i+1], x)]
        temp = torch.cat(temp, dim=-1)
        return torch.mean(weight * temp, dim=-1, keepdim=True)
    
    def visibility_function(self, x, mu, weight):
        intergrated_gaussian_func = lambda inv_sigma,mu, t: 0.5 * torch.erf(inv_sigma * (t - mu) / (2 ** 0.5)) + 0.5
        temp = []
        for i in range(weight.shape[-1]):
            temp += [intergrated_gaussian_func(2**i, mu[...,i:i+1], x)]
        temp = torch.cat(temp, dim=-1)
        return torch.mean(weight * temp, dim=-1, keepdim=True)
        
    def sigmoid_func(self, x, mu, a):

        return 1. / (1. + torch.exp(-a * (x - mu)))
    
    def integrated_sigmoid_func(self, x, mu, a, near):

        func = lambda t: t-mu + 1/(a+1e-8) * torch.log(1. + torch.exp(-a * (t-mu))) 
        res = func(x) - func(near)
        res[res < 0] = 0
        return res

    @torch.no_grad()
    def warping(self, rays_o, rays_d, samples, cam, nei_idxs, up_axis, mode):
        """
        修改这里的warping  
        samples  B x num_sample x 3
        rts     N x 3 x 4 
        nei_idxs B x num_sample x num_neighbor x 1 
        rays_o  B x 3 
        """
        
        nei_idxs = nei_idxs.squeeze(-1)
        
        B, num_sample, num_neighbor = nei_idxs.shape[:3]

        warped_samples = torch.full((B, num_sample, num_neighbor, 3), -1, dtype=torch.float32, device=self.device)
        warped_uvs = torch.full((B, num_sample, num_neighbor, 3), -1, dtype=torch.float32, device=self.device)
        warped_colors = torch.full((B, num_sample, num_neighbor, 3), 0, dtype=torch.float32, device=self.device)
        warped_colors_blur = torch.full((B, num_sample, num_neighbor, 3), 0, dtype=torch.float32, device=self.device)
        warped_mask = torch.full((B, num_sample, num_neighbor, 1), 1, dtype=torch.bool, device=self.device)
        warped_valid_mask = torch.full((B, num_sample, num_neighbor, 1), 1, dtype=torch.bool, device=self.device)
        blend_weight = torch.full((B, num_sample, num_neighbor, 1), 0, dtype=torch.float32, device=self.device)
        ray_distance = torch.full((B, num_sample, num_neighbor, 1), -1, dtype=torch.float32, device=self.device)
        nei_dir = torch.full((B, num_sample, num_neighbor, 3), 0, dtype=torch.float32, device=self.device)
        nei_bound = torch.full((B, num_sample, num_neighbor, 2), 0, dtype=torch.float32, device=self.device)
        
        sigma = self.blur_para[0]
        max_dis = self.blur_para[1]
        
        if mode == INFERENCE:
            padding = 0
        else:
            padding = self.padding
        warping_samples_cuda(rays_o, up_axis, samples, cam.get_rts(), cam.ks, self.images, self.valid_mask,
                             nei_idxs.int(), 
                             warped_samples, ray_distance, 
                             nei_bound, self.ogrid.bbox_center, self.ogrid.bbox_size,
                             nei_dir, warped_uvs, warped_colors, warped_colors_blur,
                             warped_mask, warped_valid_mask, blend_weight, sigma, 
                             max_dis, self.ogrid.near, self.ogrid.far, 
                             mode==INFERENCE, padding)

        warped_colors = warped_colors / 255. 
        warped_colors_blur = warped_colors_blur / 255.

        
        
        
        if torch.any(nei_bound == -1):
            print("Error occur!!!!!\n")

        
        # B x num_sample x num_neighbor 
        outside_viewport = (warped_uvs[...,0] < 0) | (warped_uvs[...,0] >= self.W) | (warped_uvs[...,1] < 0) | (warped_uvs[...,1] >= self.H)
        
        # uv  [-padding, W+padding] [-padding, H+padding]
        # warped_uvs[..., 0] = (warped_uvs[..., 0] + self.padding) / (self.W + 2 * self.padding)   # [0,1]
        # warped_uvs[..., 1] = (warped_uvs[..., 1] + self.padding) / (self.H + 2 * self.padding)   #[0,1]

        warped_uvs[..., 0] = (warped_uvs[..., 0].int() + self.padding) / (self.W + 2 * self.padding)   # [0,1]
        warped_uvs[..., 1] = (warped_uvs[..., 1].int() + self.padding) / (self.H + 2 * self.padding)   #[0,1]
        
        
        warped_samples = torch.cat([warped_uvs[...,:2], ray_distance], dim=-1)
        
        
        
        warped_features = None 

        # B x num_sample x num_neighbor x 1
        warped_mask = warped_mask != 0
        warped_valid_mask = warped_valid_mask != 0
        

        warped_mask = warped_mask.float()
        

        out_dict = {"warped_samples": warped_samples, "warped_colors": warped_colors,
                    "warped_mask": warped_mask, "blend_weight": blend_weight,
                    "ray_distance": ray_distance, "warped_colors_blur": warped_colors_blur,
                    "nei_dir": nei_dir, "nei_bound": nei_bound, "outside_viewport": outside_viewport}
        return edict(out_dict)


    def sample_pdf_train(self, batch_size, num_fine, inv_sigma, mu):
        """
        mu        B x 1
        inv_sigma B x 1
        return samples B x num_fine 
        """
        return torch.randn(batch_size, num_fine, device=self.device) / inv_sigma + mu 

    
    @torch.no_grad()
    def sample_pdf(self, bins, weights, N_samples, det=False, pytest=False):
        # bins = self.ray_contraction(bins)
        # Get pdf
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=self.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=self.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0., 1., N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
        
        # samples = self.inv_ray_contraction(samples)

        return samples
    
        
    def inference(self, rays_o, rays_d, z_vals, cam, mode, up_axis, ref_idxs=None):

        samples = rays_o[:,None,:] + z_vals[:,:,None] * rays_d[:,None,:]
        

        B, num_sample = z_vals.shape[:2]

        # if mode == INFERENCE:
        # tools.points2obj("sample2.obj",samples.detach().cpu().numpy().reshape(-1,3))
        # exit()
        #     input()
        
        
        # B x num_sample x num_neighbor x 1 
        # s = time.time()
        nei_idxs = self.neighbor_view_selection_v3(rays_o, rays_d, z_vals, cam, mode, up_axis, ref_idxs)
        # nei_idxs = self.neighbor_view_selection(rays_o, rays_d, samples, cam, mode, up_axis, ref_idxs)
        # torch.cuda.synchronize()
        # e = time.time()
        # t = (e - s) * 1000
        # print(f"{t}ms")
        # input()
        # print(nei_idxs)
        # exit()
        
        if mode is TRAIN:
            nei_idxs = torch.cat([ref_idxs[:,None,None,None].repeat(1, nei_idxs.shape[1], 1, 1), nei_idxs], dim=2)


        warping_out = self.warping(rays_o, rays_d, samples, cam, nei_idxs, up_axis, mode)
        warped_samples = warping_out.warped_samples
        warped_colors = warping_out.warped_colors
        mask = warping_out.warped_mask
        nei_dir = warping_out.nei_dir
        
        outside_viewport = warping_out.outside_viewport
        blend_weight = warping_out.blend_weight
        
        warped_xy = warped_samples[..., :2]
        warped_z = warped_samples[...,2:]

        
        
        
        """
        warped_xy B x num_sample x num_neighbor x 2
        mask      B x num_sample x num_neighbor x 1
        """
        
        
        # print(warped_xy.min(), warped_xy.max())
        alpha, visibility, mu, inv_sigma, z_dists, feature, gaussian_weight, warped_pred_color = \
            self.infer_(warped_xy, nei_idxs, nei_dir, 
                        warping_out.ray_distance, 
                        warping_out.nei_bound, mode)
        
        # if torch.isnan(visibility).sum() > 0:
        #     print(torch.isnan(warped_xy).sum())
        #     print(torch.isnan(warping_out.ray_distance).sum())
        #     print(torch.isnan(warping_out.nei_bound).sum())
        #     exit()

        warped_colors[outside_viewport] = warped_pred_color[outside_viewport]
        
        visibility = visibility * mask
        alpha = alpha * mask 


        out = {}


        if mode is TRAIN:
            out["ref_alpha"] = alpha[..., 0, :] 
            out["ref_visibility"] = visibility[..., 0, :] 
            out["nei_idxs"] = nei_idxs[..., 1:, :]
            out["warped_uvs"] = warped_xy[..., 1:, :]

            if self.use_ref == False:
                visibility = visibility[..., 1:, :]
                alpha = alpha[..., 1:, :]
                warped_colors = warped_colors[..., 1:, :]
                mask = mask[..., 1:, :]
                nei_dir = nei_dir[..., 1:, :]
                blend_weight = blend_weight[..., 1:, :]
                try:
                    feature = feature[..., 1:, :]
                except:
                    feature = None
                # beta = beta[..., 1:, :]
                # sharp = sharp[..., 1:,:]
                # specular = specular[..., 1:,:]
            else:
                raise NotImplementedError
                blend_weight = blend_weight[..., :-1, :]
                visibility = visibility[..., :-1, :]
                alpha = alpha[..., :-1, :]
                warped_colors = warped_colors[..., :-1, :]
                mask = mask[..., :-1, :]
                nei_dir = nei_dir[..., :-1, :]


        out.update({"visibility": visibility,
                    "alpha": alpha, "warped_colors": warped_colors, "blend_weight": blend_weight,
                    "mask": mask, "nei_dir": nei_dir, "specular": None, "feature": feature, "beta": None})
        return out

    
    def cal_weight(self, alpha, mode):
        T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1, 1), device=self.device), 1.-alpha], 1), 1)[:, :-1]
        weight = T * alpha
        # weight = torch.cat([weight[:,:-1,:], 1. - torch.sum(weight, dim=1, keepdim=True).clamp(0,1)], dim=1)
        # T = 1. - torch.cumsum(torch.cat([torch.zeros((alpha.shape[0], 1, 1), device=self.device), alpha], 1), 1)[:, :-1].clamp(0,1)
        # weight = torch.minimum(T, alpha)
        
        if mode is INFERENCE:
            sum_weight = torch.sum(weight, dim=1, keepdim=True) + 1e-8
            sum_weight[sum_weight > 0.95] = 1.0   
            weight = weight / sum_weight
            
        # weight = weight / (torch.sum(weight, dim=1, keepdim=True)+1e-8)
        return weight 
    
    def cal_ref_weight(self, alpha, visibility):
        weight = (visibility[:, :-1] - visibility[:, 1:]).clamp(0,1)
        weight = torch.cat([weight, 1. - torch.sum(weight, dim=1, keepdim=True).clamp(0,1)], dim=1)
        # weight = torch.cat([weight, visibility[:,-1:]], dim=1)
        # weight = weight / (torch.sum(weight, dim=1, keepdim=True)+1e-8)
        return weight
    
    def blending(self, rays_o, rays_d, z_vals, dists, warped_color, feature,
                 visibility, alpha, mask, nei_dir, blend_weight,
                 mode):
        """
        blend_weight B, num_sample, num_neighbor, 1
        visibility   B, num_sample, num_neighbor, 1
        alpha        B, num_sample, num_neighbor, 1
        warped_color B, num_sample, num_neighbor, 3
        mask B x num_sample x num_neighbor x 1
        tint B x num_sample x num_neighbor x 1
        """
        B, num_sample, num_neighbor = warped_color.shape[:3]
        
        

        samples = rays_o[:,None,:] + z_vals[:,:,None] * rays_d[:,None,:]


        # B x num_sample x 1
        blend_alpha = torch.sum(alpha * visibility, dim=2) / (torch.sum(visibility, dim=2)+EPS)

        weight = self.cal_weight(1. - torch.exp(-blend_alpha * dists[..., None]), mode)  
        pred_depth = torch.sum(weight * z_vals[..., None], dim=1)
        
        coeffi = self.coeffiNet((samples - self.ogrid.bbox_center.detach()) * 2.0 / self.ogrid.bbox_size.detach(), 
                               nei_dir, visibility)
        
        
        blend_color = torch.sum(warped_color * coeffi * visibility, dim=2) / (torch.sum(coeffi * visibility, dim=2)+EPS)
        pred_color = torch.sum(weight * blend_color, dim=1)
        
        out_color_bias = None
            
        out = {}
        
        out.update({"pred_color": pred_color, 
                    "pred_depth": pred_depth,
                    "contracted_samples": None,
                    "samples": samples,
                    "beta": None,
                    "bias_weight": None,
                    "blend_weight": coeffi,
                    "coeffi": coeffi,
                    "blend_color": blend_color,
                    "blend_alpha": blend_alpha,
                    "weight": weight})
        return out 
    
    
    def render_batch_rays(self, rays_o, rays_d, z_vals, dists, cam, mode, model, step, **kwargs):
        out = {}
        self.model = model 
        if mode is TRAIN:
            res = self.inference(rays_o, rays_d, z_vals, cam, mode, kwargs["up_axis"], ref_idxs=kwargs["ref_idxs"])
        else:
            res = self.inference(rays_o, rays_d, z_vals, cam, mode, kwargs["up_axis"])
        # torch.cuda.synchronize()
        # e = time.time()
        # t = (e - s) * 1000
        # print(f"inference {t}ms")
        
        out.update(res)
        visibility = res["visibility"]
        alpha = res["alpha"]
        warped_colors = res["warped_colors"]
        mask = res["mask"]
        nei_dir = res["nei_dir"]
        specular = res["specular"]
        blend_weight = res["blend_weight"]
        feature = res["feature"]


        if mode is TRAIN:
            ref_visibility = res["ref_visibility"]
            ref_alpha = res["ref_alpha"]
            out["dists"] = dists
            num_sample = z_vals.shape[1]
            out["valid"] = (mask.sum(2) > 0).sum(1) > (num_sample / 2)
            # out["valid"] = torch.ones_like(rays_o[..., :1]).bool()
            # out["bounds"] = bounds
        else:
            num_sample = z_vals.shape[1]
            out["valid"] = (mask.sum(2) > 0).sum(1) > (num_sample / 2)
            # out["valid"] = torch.ones_like(rays_o[..., :1]).bool()
            
            
        # s = time.time()
        blending_res = self.blending(rays_o, rays_d, 
                                    z_vals, dists, warped_colors, feature, 
                                    visibility, alpha, mask, nei_dir, blend_weight,
                                    mode)

        # torch.cuda.synchronize()
        # e = time.time()
        # t = (e - s) * 1000
        # print(f"blending {t}ms")
        out.update(blending_res)

        blend_color = blending_res["blend_color"]
        weight = blending_res["weight"]
        
        if mode is TRAIN:
            self.ogrid.update_value(blending_res["samples"], weight * out["valid"][:,None,:].repeat(1,num_sample, 1) , step)
                                            
        if mode is TRAIN:
            ref_weight = self.cal_ref_weight(ref_alpha, ref_visibility)
            out["ref_weight"] = ref_weight
            out["ref_visibility"] = ref_visibility[:,-1:]
            
            pred_color_ref = torch.sum(ref_weight * blend_color, dim=1)
            pred_depth_ref = torch.sum(ref_weight * z_vals[..., None], dim=1)

            out.update({"pred_color_ref":pred_color_ref, "pred_depth_ref":pred_depth_ref})
        
        return out

    def one_round_proposal(self, rays_o, rays_d, z_vals, num_proposal, dists, cam, mode, model, step, **kwargs):
        
        res = self.render_batch_rays(rays_o, rays_d, z_vals, dists, cam, mode, model, step, **kwargs)
        # s_vals = self.t_sapce_to_s(z_vals)
        proposal_z_vals = self.sample_pdf(.5 * (z_vals[...,1:] + z_vals[...,:-1]), res["weight"][:,1:-1,0], num_proposal+1, 
                                          det=mode==INFERENCE, pytest=False)
        proposal_z_vals = proposal_z_vals[:,1:]
        if mode == TRAIN:
            proposal_z_vals = proposal_z_vals.sort(dim=1)[0]
        # proposal_z_vals = self.s_space_to_t(proposal_s_vals)
        proposal_dists = self.get_dists(rays_d, proposal_z_vals)
        return res, proposal_z_vals, proposal_dists
    
    def render_rays(self, rays_o, rays_d, cam, num_sample, mode,
                    step, total_step, **kwargs):
        
        """
        rays_o B x 3 
        rays_d B x 3
        nei_idxs B x num_neighbor  
        return color B x 3
        """
        out = {}

        num_sample = 192 - int(64 * min(step / 20000, 1.0))
        z_vals, dists = self.ogrid.sample_points_LLFF(rays_o, rays_d, num_sample, coarse_num=64)
            

        fine_out = self.render_batch_rays(rays_o, rays_d, z_vals, dists, cam, mode, self.fine_model, step, **kwargs)
        out.update(fine_out)
        
        return out
            