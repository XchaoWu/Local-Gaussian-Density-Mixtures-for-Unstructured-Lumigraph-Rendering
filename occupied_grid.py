import numpy as np 
import torch  
import math
import os 
from cuda import (
    update_grid_value_cuda,
    sample_points_grid,
    ray_aabb_intersection,
    voxelize_mesh,
    get_focus_cuda,
    set_all_cuda,
    update_voxel_grid_cuda,
    voxel_grid_to_mask_cuda,
    split_voxel_grid_cuda,
    sparse_voxel_contracted_grid_sampling_cuda)

from tools import tools 
from cfg import * 
from tqdm import tqdm 

"""
Data Types:

1. only near far   NeRF LLFF dataset 
2. with initial mesh   
3. initial by mask   DTU dataset 

"""
class OccupiedGrid:
    def __init__(self, log2dim_resolution, device):
        """
        log2dim_resolution  
        """  
        
        self.device = device 
        self.thresh = 0.01
        
        self.log2dim_resolution = torch.tensor([log2dim_resolution[0],
                                                log2dim_resolution[1],
                                                log2dim_resolution[2]], dtype=torch.int32)
        self.log2dim_resolution_gpu = self.log2dim_resolution.to(self.device)
        # size_data = int(math.ceil(2**torch.sum(self.log2dim_resolution) / 64))
        
        # self.grid_data = torch.zeros((size_data,), dtype=torch.int64, device=self.device)
        
        # self.grid_value = torch.ones((2**log2dim_resolution[0], 2**log2dim_resolution[1], 2**log2dim_resolution[2]),
        #                               dtype=torch.float32, device=self.device)

        self.grid_value = torch.zeros((2**log2dim_resolution[0], 2**log2dim_resolution[1], 2**log2dim_resolution[2]),
                                      dtype=torch.float32, device=self.device)
        
        # set_all_cuda(self.grid_data, *self.log2dim_resolution, True)
        self.update_mask()
        
        self.near = None
        self.far = None
        
        self.inv = False
        # self.INF_FAR = 1e6
        # # for very far background
        # self.use_INF_FAR = False
        
    def load(self, path):
        raise NotImplementedError
        # grid_mask = np.load(path)
        # self.grid_mask = torch.from_numpy(grid_mask).to(self.device)
        # assert self.grid_mask.shape[0] == self.resolution[0]
        # assert self.grid_mask.shape[1] == self.resolution[1]
        # assert self.grid_mask.shape[2] == self.resolution[2]
        # print("load grid successfully!")
        
    def vis_grid(self, path):
        print("export sparse voxel grid ... ")
        resolution = 2 ** self.log2dim_resolution
        grid_size = self.bbox_size.cpu() / resolution
        # mask = torch.full((resolution[0],resolution[1],resolution[2]), 0, 
        #                   dtype=torch.bool, device=self.device)
        # voxel_grid_to_mask_cuda(self.grid_data, *self.log2dim_resolution, mask)
        X,Y,Z = torch.meshgrid(torch.arange(0, resolution[0], 1),
                               torch.arange(0, resolution[1], 1),
                               torch.arange(0, resolution[2], 1))
        centers = torch.stack([X,Y,Z], -1).reshape(-1,3) * grid_size + grid_size / 2. 
        centers = centers[self.grid_mask.cpu().reshape(-1,)] + self.bbox_corner.cpu()
        size = torch.ones_like(centers) * grid_size
        if centers.shape[0] <= 0:
            print("no grid to export")
            return 
        vertices, faces = tools.draw_AABB(centers.detach().cpu().numpy(), size.detach().cpu().numpy())
        tools.mesh2obj(path, vertices, faces)


    
    def set(self, bbox_center, bbox_size):
        self.bbox_center = torch.from_numpy(bbox_center).to(self.device)
        self.bbox_size = torch.from_numpy(bbox_size).to(self.device)
        self.bbox_corner = self.bbox_center - self.bbox_size / 2.
        print("set bbox corner & bbox size\n")
        print("bbox_center", self.bbox_center, "bbox_size", self.bbox_size)
    
    def split(self, max_log2dim=8):
        
        # if torch.any(self.log2dim_resolution > max_log2dim):
        #     print("already max, no split")
        #     return 
        old = self.log2dim_resolution.clone()
        self.log2dim_resolution += 1
        self.log2dim_resolution[self.log2dim_resolution > max_log2dim] = max_log2dim
        scale = 2 ** (self.log2dim_resolution - old).to(self.device)
        
        self.grid_value = self.grid_value.repeat_interleave(scale[0], dim=0).repeat_interleave(scale[1], dim=1).repeat_interleave(scale[2], dim=2)
        self.update_mask()


    @torch.no_grad()
    def clear(self):
        self.grid_value = torch.zeros((2**log2dim_resolution[0], 2**log2dim_resolution[1], log2dim_resolution[2]),
                                      dtype=torch.float32, device=self.device)
        self.update_mask()
        
        # size_data = int(math.ceil(2**torch.sum(self.log2dim_resolution) / 64))
        # self.grid_data = torch.zeros((size_data,), dtype=torch.int64, device=self.device)
    
    @torch.no_grad()
    def update_value(self, pts, value, step, gamma=0.9):
        """
        pts   ... x 3 
        value ... x 1 
        """

        pts = pts.reshape(-1,3)
        value = value.reshape(-1,1)
        
        # B x 3 
        pts = (pts - self.bbox_corner) / self.bbox_size
        
        
        update_grid_value_cuda(self.grid_value, *self.log2dim_resolution, gamma, pts, value)
        self.update_mask()
        
        # contract space point         
        # update_voxel_grid_cuda(pts.reshape(-1,3), value.reshape(-1,1),
        #                        self.grid_data, *self.log2dim_resolution, 
        #                        self.thresh)
    @torch.no_grad()
    def update_mask(self):
        self.grid_mask = self.grid_value > self.thresh


        
    def remove_invisiable(self, camera):
        # raise NotImplementedError
        temp_mask = torch.zeros_like(self.grid_mask)
        print("pruning invisible ... ")
        for i in tqdm(range(camera.num_camera)):
            rays_o, rays_d, _  = camera.get_rays(view_idx=i)
            rays_o = rays_o.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)
            get_focus_cuda(rays_o, rays_d, temp_mask, *self.log2dim_resolution,
                           self.bbox_corner, self.bbox_size, self.near, self.far)
        self.grid_value[~temp_mask] = 0.0
        self.update_mask()
        
    def get_bbox(self):
        return self.bbox_center, torch.ones_like(self.bbox_center) * self.bbox_size


    def init_with_mask(self, masks, camera):
        raise NotImplementedError

        
        
    
    @torch.no_grad()
    def init_scene_360(self, near, far, camera):
        
        self.near = near
        self.far = far 
        
        self.disth = 2.0 - near
        # self.disth = 2.0 
        # self.set(np.array([0, 0, 0,], dtype=np.float32), 2 * np.ones((3,), dtype=np.float32))
        # self.set(np.array([0, 0, 0,], dtype=np.float32), (far - near) * np.ones((3,), dtype=np.float32))
        
        self.set(np.array([0, 0, 0,], dtype=np.float32), 2 * (far - near) * np.ones((3,), dtype=np.float32))
        return
      

    def init_with_mesh(self, mesh_dir):
        raise NotImplementedError
        # self.grid_mask = torch.zeros((self.resolution[0], self.resolution[1], self.resolution[2]),
        #                                  dtype=torch.bool)
        # voxelize_mesh(torch.tensor([self.log2dim_resolution,self.log2dim_resolution,self.log2dim_resolution]).int(), 
        #               self.bbox_corner.cpu(), 
        #               torch.tensor([self.bbox_size,self.bbox_size,self.bbox_size]).float(),
        #               mesh_dir, 
        #               self.grid_mask, True)
        # self.grid_mask = self.grid_mask.to(self.device)
    
    def init_with_blender(self, c2ws):
        raise NotImplementedError
        # 3 
        # self.bbox_center = torch.tensor([0.,0., 1.], dtype=torch.float32, device=c2ws.device)
        # self.bbox_size = torch.tensor([5.,5.,5.], dtype=torch.float32, device=c2ws.device)
        # self.bbox_corner = self.bbox_center - self.bbox_size / 2.
        # print("bbox_center", self.bbox_center, "bbox_size", self.bbox_size)
        
    def init_with_nearfar(self, camera, near, far):        
        
        """
        
        [NOTE] Only for forward facing data (facing along z axis in world space)
        
        """
        # raise NotImplementedError
        
        # for forward facing LLFF 
    
        self.near = near
        self.far = far 

        pts = []
        for i in tqdm(range(camera.num_camera)):
            rays_o, rays_d, _  = camera.get_rays(view_idx=i)
            rays_o = rays_o.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)
            
            # B x 3 
            near_pts = rays_o + near * rays_d 
            far_pts = rays_o + far * rays_d
            
            temp_pts = torch.cat([near_pts, far_pts], dim=0)
            
            # print(far_pts)
            # input
            
            pts += [temp_pts.min(dim=0)[0]]
            pts += [temp_pts.max(dim=0)[0]]
            
        pts = torch.stack(pts, 0)
        # print(pts)
        
            
        min_pts = torch.min(pts, dim=0)[0]
        max_pts = torch.max(pts, dim=0)[0]
        
        size = (max_pts - min_pts).max() * 1.2
        
        # self.bbox_size = (max_pts - min_pts)
        
        # make sure xy plane is big enough, z axis is forward 
        # self.bbox_size[:2] *= 1.2
        self.bbox_size = size.repeat(3)
        
        self.bbox_center = (max_pts + min_pts) / 2.
        
        self.bbox_corner = self.bbox_center - self.bbox_size / 2.
        
        print("bbox_center", self.bbox_center, "bbox_size", self.bbox_size)


    def init_grid_cover_center(self, camera, near, far):
        """
        Grid will only cover the center, for data like ouside-in
        """   
        self.near = near
        self.far = far 
        # num_camera x 3
        camera_centers = camera.get_poses()[:,:,3]
        min_pts = torch.min(camera_centers, dim=0)[0]
        max_pts = torch.max(camera_centers, dim=0)[0]
        
        size = (max_pts - min_pts).max() * 1.2
        self.bbox_size = size.repeat(3)
        self.bbox_center = (max_pts + min_pts) / 2.
        self.bbox_corner = self.bbox_center - self.bbox_size / 2.
        print("bbox_center", self.bbox_center, "bbox_size", self.bbox_size)
        
        
    @torch.no_grad()
    def sample_points(self, rays_o, rays_d, num_sample, sample_mode, inv=False):
        
        if sample_mode == GRID:
            return self.sparse_grid_sampling(rays_o, rays_d, num_sample)
        
        elif sample_mode == NEAR_FAR:
            return self.near_far_grid_sampling(rays_o, rays_d, num_sample, self.inv)
    
    @torch.no_grad()
    def disparity_sampling(self, near, far):
        """
        sampling in disparity
        near B x 1 or scalar
        far B x 1 or scalar
        """
        z_vals = torch.linspace(0., 1., steps=num_sample, device=self.device)
        z_vals = 1./(1./near * (1.-z_vals[None,:]) + 1./ far * (z_vals[None,:]))
        return z_vals

    @torch.no_grad()
    def near_far_sampling(self, rays_o, rays_d, num_sample, inv=True, near=None, far=None):
        
        if near == None or far == None:
            bounds = torch.full((rays_o.shape[0], 2), -1, dtype=torch.float32, device=self.device)
            ray_aabb_intersection(rays_o, rays_d, 
                                self.bbox_center, self.bbox_size, bounds)
        else:
            bounds = None
        if near == None:
            near = bounds[:,0:1]

        if far == None:
            far = bounds[:,1:2]
            # far[far < near] = near 
        
        z_vals = torch.linspace(0., 1., steps=num_sample, device=self.device)    
        
        # print(near.shape, far.shape)
 
        if inv:
            z_vals = 1./(1./near * (1.-z_vals[None,:]) + 1./ far * (z_vals[None,:]))
        else:
            z_vals = z_vals[None,:] * (far - near) + near  
            
        if z_vals.shape[0] == 1:
            z_vals = z_vals.repeat(rays_o.shape[0], 1)

        dists = torch.cat([z_vals[...,1:] - z_vals[...,:-1], 1e10*torch.ones(rays_o.shape[0], 1, device=self.device)], dim=-1)
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        return z_vals, dists, bounds
    
    @torch.no_grad()
    def near_far_grid_sampling(self, rays_o, rays_d, num_sample, inv=True):
        bounds = torch.full((rays_o.shape[0], 2), -1, dtype=torch.float32, device=self.device)
        ray_aabb_intersection(rays_o, rays_d, self.bbox_center, self.bbox_size, bounds)
        z_vals = torch.linspace(0., 1., steps=num_sample, device=self.device)
        # bounds[:,:1] = bounds[:,:1] + 0.1

        # far = bounds[:,1:]
        # far = self.far
        far = torch.minimum(bounds[:,1:], self.far * torch.ones_like(bounds[:,1:]))
        near = torch.maximum(bounds[:,0:1], self.near * torch.ones_like(bounds[:,0:1]))
        
        
        if inv:
            z_vals = 1./(1./near * (1.-z_vals[None,:]) + 1./ far * (z_vals[None,:]))
        else:
            z_vals = z_vals[None,:] * (far - near) + near  
        
        dists = torch.cat([z_vals[...,1:] - z_vals[...,:-1], 1e10*torch.ones(rays_o.shape[0], 1, device=self.device)], dim=-1)
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        return z_vals, dists, bounds
    
    @torch.no_grad()
    def sparse_grid_sampling(self, rays_o, rays_d, num_sample):
        """
        rays_o    B x 3
        rays_d    B x 3
        """
        bounds = torch.full((rays_o.shape[0], 2), -1, dtype=torch.float32, device=self.device)
        ray_aabb_intersection(rays_o, rays_d, 
                            self.bbox_center, self.bbox_size, bounds)
            
        z_vals = torch.full((rays_o.shape[0], num_sample), -1, dtype=torch.float32, device=self.device)
        dists = torch.full((rays_o.shape[0], num_sample), 0.0, dtype=torch.float32, device=self.device)
        sample_points_grid(rays_o, rays_d, z_vals, dists, self.bbox_corner, self.bbox_size,
                        self.grid_mask, self.near, self.far, *self.log2dim_resolution)
        
        # print(z_vals[0].cpu().numpy().tolist())
        # print(dists[0].cpu().numpy().tolist())
        # input()
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        
        return z_vals, dists, bounds
    
        
    @torch.no_grad()
    def sample_points_360(self, rays_o, rays_d, num_sample, ratio=3):
        coarse_num = int(num_sample // ratio)
        fine_num = num_sample - coarse_num
        
        z_vals_fg, z_dist_fine, bounds = self.sample_points(rays_o, rays_d, fine_num, GRID)
        # 有grid相交的光线
        mask = torch.all(z_vals_fg!=-1, dim=-1)
        uniform_mask = ~mask
        if uniform_mask.sum() > 0:
            z_vals_uniform, _, _ = self.near_far_sampling(rays_o[uniform_mask], rays_d[uniform_mask], fine_num, 
                                                         inv=False, near=None, far=None)
            z_vals_fg[uniform_mask] = z_vals_uniform
        
        
        z_vals_disparity, _, _  = self.near_far_grid_sampling(rays_o, rays_d, coarse_num, inv=True)
        z_vals, _ = torch.cat([z_vals_fg, z_vals_disparity], dim=-1).sort(dim=-1)

        dists = torch.cat([z_vals[...,1:] - z_vals[...,:-1], 1e10*torch.ones(rays_o.shape[0], 1, device=self.device)], dim=-1)
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        
        return z_vals, dists
        
    @torch.no_grad()
    def sample_points_LLFF(self, rays_o, rays_d, num_sample, coarse_num=64):
        
        # return self.near_far_sampling(rays_o, rays_d, num_sample, inv=True, near=self.near, far=self.far)
        fine_num = num_sample - coarse_num

        # coarse_num = 32 
        # fine_num = 160

        z_vals = torch.full((rays_o.shape[0], num_sample), -1, dtype=torch.float32, device=self.device)
        
        z_vals_fine, z_dist_fine, bounds = self.sample_points(rays_o, rays_d, fine_num, GRID)
        # 有grid相交的光线
        mask = torch.all(z_vals_fine!=-1, dim=-1)
        coarse_mask = ~mask
        
        if coarse_mask.sum() > 0:
            z_vals_coarse, _, _ = \
                self.sample_points(rays_o[coarse_mask], rays_d[coarse_mask], num_sample, NEAR_FAR)
            z_vals[coarse_mask] = z_vals_coarse
        
        if mask.sum() > 0 and coarse_num > 0:
            z_vals_coarse, _, _ = \
                self.sample_points(rays_o[mask], rays_d[mask], coarse_num, NEAR_FAR)
            
            z_vals_fine, _ = torch.cat([z_vals_coarse, z_vals_fine[mask]], dim=-1).sort(dim=-1)
            z_vals[mask] = z_vals_fine

        dists = torch.cat([z_vals[...,1:] - z_vals[...,:-1], 1e10*torch.ones(rays_o.shape[0], 1, device=self.device)], dim=-1)
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        
        return z_vals, dists

    @torch.no_grad()
    def sparse_voxel_grid_sampling(self, rays_o, rays_d, num_sample):
        
        """ if hit the grid, sample num = 192 
        """

        z_vals, dists, bounds = self.sample_points(rays_o, rays_d, num_sample, GRID)
        # 有grid相交的光线
        mask = torch.all(z_vals!=-1, dim=-1)

        coarse_mask = ~mask
        
        if coarse_mask.sum() > 0:
            z_vals_coarse, dists_coarse, _ = \
                self.sample_points(rays_o[coarse_mask], rays_d[coarse_mask], num_sample, NEAR_FAR, inv=False)
            z_vals[coarse_mask] = z_vals_coarse
            dists[coarse_mask] = dists_coarse
    
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        # dists = torch.cat([z_vals[...,1:] - z_vals[...,:-1], 1e10*torch.ones(rays_o.shape[0], 1, device=self.device)], dim=-1)
        # dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        
        return z_vals, dists
    
    
if __name__ == "__main__":
    import random 
    from tqdm import tqdm 
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


    device = torch.device("cuda:0")
    bbox_center = np.array([0,0,0],dtype=np.float32)
    bbox_size = np.array([4.0,4.0,4.0],dtype=np.float32)
    
    grid = OccupiedGrid((5,5,5), device)
    grid.set(bbox_center, bbox_size)
    grid.near = 0.1
    grid.far = 2.0
    
    pts = torch.tensor([0.2,0.2,0.2, 0.9,0.9,0.9], dtype=torch.float32, device=device).reshape(-1,3)
    value = torch.ones_like(pts[...,:1])
    grid.update_value(pts, value, 1)

    
    grid.vis_grid("/data/xchao/data/grid.obj")
    
    
    rays_o = torch.zeros((1,3),dtype=torch.float32, device=device)
    rays_d = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device).reshape(-1,3)
    
    z_vals, dists, empty = grid.sparse_grid_sampling(rays_o, rays_d, 192)
    
    print(z_vals)
    samples = rays_o[:,None,:] + z_vals[...,None] * rays_d[:,None,:]
    # print(samples)
    # exit()
    
    from tools import tools 
    tools.points2obj("/data/xchao/data/samples.obj",samples.cpu().numpy().reshape(-1,3))
    
#     # print(grid.grid_value)
    
#     # pts = torch.tensor([0.1, 0.1, 0.1, 0.2,0.1,0.1, 0.3, 0.3, 0.3], device=device).float().reshape(-1,3)
#     # value = torch.tensor([0.6, 0.3, 0.7],device=device).float()
    
#     # grid.update_value(pts, value, alpha=0.9)
#     # print(grid.grid_value)