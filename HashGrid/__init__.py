import torch 
import torch.nn as nn 
import numpy as np 
from .PyHashGrid import PyHashGrid
from .PyHashGridBG import PyHashGridBG
from .PyHashGrid2D import HashGrid2D
# from .function import ContractBGSpace
from .lib.HASHGRID import Sampler
from .lib.HASHGRID import (
    ray_block_intersection, 
    sample_points, 
    prepare_points, 
    sort_by_key, 
    pts_inference, 
    accumulate_color,
    ray_firsthit_block,
    inverse_z_sampling,
    bg_pts_inference,
    get_last_block,
    update_outgoing_bidx,
    update_outgoing_bidx_v2,
    bg_pts_inference_v2,
    process_occupied_grid)
# from .hashencoder import HashEncoder
import sys,os
sys.path += ['../', './']
from cuda import ray_aabb_intersection, sample_points_contract
from cuda import voxelize_mesh,sample_points_grid
from cfg import * 
from tools import tools
from tqdm import tqdm 

class HashGrid(nn.Module):
    def __init__(self, device, bbox_corner, bbox_size, 
                 log2_hashmap_size = 24,
                 grid_resolution = [32,2048], 
                 sampler_log2dim=4, 
                 init_outside=False,
                 model_path=""):
        super(HashGrid, self).__init__()
        """
        bbox_center 3
        bbox_size 3 
        """
        self.device = device
        self.bbox_center = bbox_corner + bbox_size / 2. 
        # self.bbox_size = torch.tensor([float(bbox_size)], device=self.device, dtype=torch.float32).repeat(3)
        
        # [NOTE] 2 times for backgournd 
        self.bbox_size = bbox_size * 2 
        self.min_bbox = self.bbox_center - self.bbox_size / 2.
        self.max_bbox = self.bbox_center + self.bbox_size / 2.
        self.log2_hashmap_size = log2_hashmap_size
        
        # 3 
        self.finest_resolution = (self.bbox_size / self.bbox_size.min() * grid_resolution[1]).int().cpu()
        self.base_resolution = (self.bbox_size / self.bbox_size.min() * grid_resolution[0]).int().cpu()

        # self.finest_resolution = 2048

        self.HE = PyHashGridBG(self.device,  self.min_bbox, self.bbox_size,
                               n_levels=16, n_features_per_level=2,
                               log2_hashmap_size=self.log2_hashmap_size, 
                               base_resolution=self.base_resolution,
                               finest_resolution=self.finest_resolution, init_mode="xavier").to(device)
        
        
        self.sampler = Sampler()
        self.last_sampler_log2dim = sampler_log2dim
        self.sampler_log2dim = sampler_log2dim - torch.log2(self.bbox_size.max() / self.bbox_size).int()
        # self.sampler_log2dim = sampler_log2dim + (self.bbox_size / self.bbox_size.min()).int() - 1
        self.occupied_grid = torch.zeros((2**self.sampler_log2dim[0],
                                            2**self.sampler_log2dim[1],
                                            2**self.sampler_log2dim[2]), dtype=torch.bool)
        # print("FIXME HERE SAMPLER!!!!!")
        min_bbox = self.min_bbox + self.bbox_size / 4. 
        bbox_size = self.bbox_size / 2. 
        self.outside = torch.zeros_like(self.occupied_grid)
        # init_outside = True # for all 
        voxelize_mesh(self.sampler_log2dim.cpu(), min_bbox.cpu(), bbox_size.cpu(), model_path, self.occupied_grid, init_outside, self.outside)
        # self.sampler.build(self.sampler_log2dim, min_bbox.cpu(), bbox_size.cpu(), model_path, self.occupied_grid,
        #                    init_outside)
        self.occupied_grid = self.occupied_grid.to(self.device)
        self.outside = self.outside.to(self.device)
        self.grid_resolution = torch.tensor([2**self.sampler_log2dim[0],
                                            2**self.sampler_log2dim[1],
                                            2**self.sampler_log2dim[2]], dtype=torch.int32, device=self.device)

        
    def vis_gird(self, path, bg=False):

        if bg:
            return 
        else:
            sampler_log2dim = self.sampler_log2dim.cpu()
            grid = self.occupied_grid.cpu()
            name = "grid.obj"
            min_bbox = self.min_bbox + self.bbox_size / 4. 
            bbox_size = self.bbox_size / 2. 

        grid_size = bbox_size.cpu() / (2**sampler_log2dim)

        X,Y,Z = torch.meshgrid(torch.arange(0, 2**sampler_log2dim[0], 1),
                               torch.arange(0, 2**sampler_log2dim[1], 1),
                               torch.arange(0, 2**sampler_log2dim[2], 1))
        centers = torch.stack([X,Y,Z], -1).reshape(-1,3) * grid_size + grid_size / 2. 
        centers = centers[grid.reshape(-1)] + min_bbox.cpu()
        size = torch.ones_like(centers) * grid_size
        vertices, faces = tools.draw_AABB(centers.cpu().numpy(), size.cpu().numpy())
        tools.mesh2obj(os.path.join(path,name), vertices, faces)

    def generateMeshGrid(self, step, max_res, device='cpu', ox=0, oy=0, oz=0):
        X,Y,Z = torch.meshgrid(torch.arange(0, max_res[0], step, device=device),
                               torch.arange(0, max_res[1], step, device=device),
                               torch.arange(0, max_res[2], step, device=device))
        return torch.stack([X+ox,Y+oy,Z+oz], -1).reshape(-1,3)


    @torch.no_grad()
    def pruning_block_grid(self, global_step, decoder, sub_split=False, pruning_th=0.4, batch_size=128**3):
        """
        log2dim 
        coarse to fine
        """
        # assert log2dim >= self.sampler_log2dim

        # 设定目标 grid log2dim 
        if sub_split == False:
            log2dim = self.sampler_log2dim
            scale = 1
        else:
            log2dim = self.sampler_log2dim + 1 
            scale = 2
        self.new_occupied_grid = torch.zeros((2**log2dim[0], 2**log2dim[1], 2**log2dim[2]), 
                                             dtype=torch.bool, device=self.device)
        # (3,) 目标 grid 分辨率
        grid_resolution = (2 ** log2dim).to(self.device)

        # 每个grid 内部需要的分辨率, 总分辨率有一半是分给background的, shape (3,)
        total_res = self.finest_resolution
        sample_resolution = (total_res / 2. ).to(self.device) / grid_resolution
        sample_resolution = sample_resolution.int()

        # voxel_size = torch.mean(self.bbox_size / 2. / grid_resolution / sample_resolution)
        # sample_resolution[sample_resolution > 64] = 64

        #
        # voxel_size = self.bbox_size.cpu() / pruning_density
        
        # self.outside = self.outside.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
        # sparse点
        xs,ys,zs = torch.where(self.occupied_grid.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)==True)
        locs = torch.stack([xs, ys, zs], -1).long()

        # N x 3 [0,1] 占 bbox_size / 2. 比例
        grid_corner = locs / grid_resolution

        num_grid = grid_corner.shape[0]

        # (HWD) x 3 每个 grid 内部需要采样的点  占 bbox_size / 2. 比例
        grid_point = self.generateMeshGrid(1, sample_resolution, device=self.device) / (sample_resolution * grid_resolution)
        
        run_num_grid = max(int(batch_size / torch.prod(sample_resolution)), 1)
        # stand_sample = 512 
        # run_num_grid = max(int(stand_sample / sample_resolution.min()), 1)
        # grid_corner = grid_corner.to(self.device)
        # grid_point = grid_point.to(self.device)
        # locs = locs.to(self.device)
        alpha_res = torch.zeros_like(grid_corner[...,0])
        for i in tqdm(range(0,num_grid,run_num_grid), desc="pruning"):
            pts = (grid_corner[i:i+run_num_grid,None,:] + grid_point[None, ...]) * 2 - 1 
            actual_num = pts.shape[0]
            features = self.HE(pts.reshape(-1,3)) * self.weight_feature(global_step)[None,:].repeat_interleave(2, dim=-1)
            # B x 3 
            alpha = 1 - torch.exp(-1.0 * decoder.inference_sigma(features))
            max_alpha, _ = torch.max(alpha.reshape(actual_num, -1), dim=-1)
            alpha_res[i:i+actual_num] = max_alpha
        
        locs = locs[alpha_res > pruning_th]
        self.new_occupied_grid[locs[:, 0],locs[:,1],locs[:,2]] = True  
        # pts = pts[:10]
        # from tools import tools 
        # tools.points2obj(f"/data/wxc/data/sig23/coffee/sample.obj", pts.detach().cpu().numpy().reshape(-1,3))

        self.sampler_log2dim = log2dim
        self.occupied_grid = self.new_occupied_grid
        self.grid_resolution = torch.tensor([2**self.sampler_log2dim[0],
                                            2**self.sampler_log2dim[1],
                                            2**self.sampler_log2dim[2]], dtype=torch.int32, device=self.device)
        # self.sampler.rebuild(self.occupied_grid, self.sampler_log2dim)
        print(f"finished pruning resolution: {grid_resolution}x{grid_resolution}x{grid_resolution} occupied: {int(torch.sum(self.occupied_grid))}")
    
    @torch.no_grad()
    def pruning_grid(self, global_step, decoder, log2dim, pruning_th):
        assert log2dim >= self.last_sampler_log2dim, f"log2dim {log2dim} last_sampler_log2dim {self.last_sampler_log2dim}"
        if log2dim == self.last_sampler_log2dim:
            sub_split = False 
        else:
            sub_split = True
            self.last_sampler_log2dim = self.last_sampler_log2dim + 1

        self.pruning_block_grid(global_step, decoder, sub_split=sub_split, pruning_th=pruning_th)


    def weight_feature(self, global_step):
        # alpha = max(min(global_step / 5000 * 15 + 1, 16), 0)

        # alpha = 1
        alpha = 16
        k = torch.arange(16,dtype=torch.float32,device=self.device)
        weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        return weight
    
    # def weight_bg_feature(self, ratio):
    #     """
    #     B x 1 
    #     """
    #     alpha = torch.clamp(ratio * 8 + 8, 0, 16)
    #     # alpha = max(min(ratio * 8 + 8, 16), 0)
    #     # alpha = 2
    #     k = torch.arange(16,dtype=torch.float32,device=self.device)
    #     weight = (1-(alpha-k[None,...]).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
    #     return weight.repeat_interleave(2, dim=-1)
    
    def export(self, path):
        features = self.HE.features.detach().cpu().numpy().astype(np.float16)
        occupied_grid = self.occupied_grid.detach().cpu()
        np.savez(os.path.join(path, "feature.npz"), 
                        features=features, 
                        occupied_grid=occupied_grid,
                        block_corner=self.min_bbox.cpu().numpy(), 
                        block_size=self.bbox_size.cpu().numpy(),
                        grid_log2dim=self.sampler_log2dim.cpu().numpy(),
                        resolution=self.HE.resolution.cpu().numpy())
    
    def load(self, path):
        file = np.load(os.path.join(path, "feature.npz"))
        self.HE.features = nn.Parameter(torch.from_numpy(file["features"]).float().to(self.device))
        self.occupied_grid = torch.from_numpy(file["occupied_grid"]).to(self.device)
        self.block_corner = torch.from_numpy(file["block_corner"]).to(self.device)
        self.block_size = torch.from_numpy(file["block_size"]).to(self.device)
        self.sampler_log2dim = torch.from_numpy(file["grid_log2dim"]).to(self.device)
        self.HE.resolution = torch.from_numpy(file["resolution"]).to(self.device)


    def toCPU(self):
        self.HE.resolution = self.HE.resolution.cpu()
        print("Download HashGrid to CPU")

    def toGPU(self):
        self.HE.resolution = self.HE.resolution.to(self.device)
        print("Upload HashGrid to GPU!")
    
    # def samplePoints(self, rays_o, rays_d, num_sample):
    #     z_vals = torch.full((rays_o.shape[0], num_sample), -1, dtype=torch.float32, device=self.device)
    #     self.sampler.samplePoints(rays_o, rays_d, num_sample, z_vals)
    #     dists = z_vals[:,1:] - z_vals[:,:-1]
    #     dists = torch.cat([dists, 1e-6*torch.ones(dists[...,:1].shape, device=rays_o.device)], -1)
    #     return z_vals, dists
    
    def samplePoints(self, rays_o, rays_d, num_sample):
        z_vals = torch.full((rays_o.shape[0], num_sample), -1, dtype=torch.float32, device=self.device)
        dists = torch.full((rays_o.shape[0], num_sample), -1, dtype=torch.float32, device=self.device)
        sample_points_grid(rays_o, rays_d, z_vals, dists, 
                             self.min_bbox + self.bbox_size / 4.,
                             self.bbox_size / 2., 
                             self.occupied_grid, self.sampler_log2dim)
        return z_vals, dists
    
    def invalid_sampling_underground(self, rays_o, rays_d, bound):
        # B x 3
        outgoing_point = rays_o + bound[:,1:] * rays_d
        # 3 
        bbox_corner = self.bbox_center - self.bbox_size / 4.

        return ~(torch.abs(outgoing_point[:,1] - bbox_corner[1]) < 0.0001)

    # Hierarchical sampling (section 5.2)
    def sample_pdf(self, bins, weights, N_samples):
        # Get pdf
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=self.device)
        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
        return samples

    @torch.no_grad()
    def invalid_sampling_outsidemesh(self, rays_o, rays_d, bound, fmesh):
        valid = (bounds[:, 1] != -1) 
        rays_o[valid] = rays_o[valid] + bounds[valid,1:] * rays_d[valid]
        depth = torch.zeros_like(rays_o[...,:1])
        fmesh.fisrtHit(rays_o, rays_d, depth)
        return depth[...,0] > 0

    @torch.no_grad()
    def background_sampling(self, fmesh, rays_o, rays_d, num_samples):
        z_vals, valid = fmesh.background_sampling(rays_o, rays_d, num_samples,
                                                  float(self.bbox_size.cpu().max()) / 10)
        dists = z_vals[:,1:] - z_vals[:,:-1]
        dists = torch.cat([dists, 1e-6*torch.ones(dists[...,:1].shape, device=rays_o.device)], -1)

        return z_vals, dists, valid 
    
    @torch.no_grad()
    def inverse_z_sampling(self, rays_o, rays_d, num_sample, invalid_underground=True):

        bounds = torch.full((rays_o.shape[0], 2), -1, dtype=torch.float32, device=rays_o.device)
        ray_aabb_intersection(rays_o, rays_d, self.bbox_center, self.bbox_size/2., bounds)
        # valid = torch.all(bounds!=-1, dim=-1)
        # disable sampling under ground 
        # valid = valid & self.invalid_sampling_underground(rays_o, rays_d, bounds)
        if invalid_underground:
            valid = self.invalid_sampling_underground(rays_o, rays_d, bounds)
        else:
            valid = torch.ones_like(rays_d[...,0]).bool() 

        bounds[torch.any(bounds==-1, dim=-1),1:] = 0.1
        # print(valid.shape)
        # exit()
        
        t_vals = torch.linspace(0., 1., steps=num_sample, device=self.device)[None,:]

        # [FIXME] far here 
        z_vals = 1./(1./(bounds[:,1:]+1e-6) * (1.-t_vals) + 1./ 1e6 * (t_vals))
        z_vals = z_vals.expand([rays_o.shape[0], num_sample])


        dists = z_vals[:,1:] - z_vals[:,:-1]
        dists = torch.cat([dists, 1e-6*torch.ones(dists[...,:1].shape, device=rays_o.device)], -1)

        return z_vals, dists, valid

    def contract2euclidean(self, x):
        """
        x ... x 3
        """
        abs_x = torch.abs(x)

        # ... x 1
        max_abs_x, _  = torch.max(abs_x, dim=-1, keepdim=True)
        L_infinity_norm = 1. / (2 - max_abs_x) 

        out_x = torch.where(abs_x == max_abs_x, L_infinity_norm * torch.sign(x), x * L_infinity_norm)
        return (out_x + 2) / 4. * self.bbox_size + self.min_bbox
    
    def sample_points_contract_space(self, rays_o, rays_d, num_sample):
        contract_z_vals, valid, contract_rays_o, contract_rays_d = self.contract_bg_sampling(rays_o, rays_d, num_sample)

        # contract space 
        contract_x = contract_rays_o[valid, None, :] + contract_z_vals[valid, :, None] * contract_rays_d[valid, None, :]

        z_vals = torch.full((rays_o.shape[0], num_sample), -1, dtype=torch.float32, device=rays_o.device)
        with torch.no_grad():
            # B x N x 3 
            x = self.contract2euclidean(contract_x)
            # B x N 
            z_vals[valid] = torch.norm(x - rays_o[valid, None, :], dim=-1) / torch.norm(rays_d[valid], dim=-1, keepdim=True)

        # 控制，防止 dist为负数
        dists = z_vals[...,1:] - z_vals[...,:-1]
        valid = valid & torch.all(dists >= 0, dim=-1)

        return z_vals, valid

    def recover_depth_from_contract(self, contract_depth, contract_rays_o, contract_rays_d, 
                                    rays_o, rays_d):
        """
        """
        # B x 3 
        contract_x = contract_rays_o + contract_depth * contract_rays_d

        # [-2, 2]
        x = self.contract2euclidean(contract_x)
        # B x 1
        return torch.norm(x - rays_o, dim=-1, keepdim=True) / torch.norm(rays_d, dim=-1, keepdim=True)

    @torch.no_grad()
    def contract_bg_sampling(self, rays_o, rays_d, num_sample):
        """
        给定 欧式空间的 rays_o, rays_d
        返回 contract 空间的 z_vals 和 valid 
        """
        #STEP 1  计算 contract space 的 rays_o rays_d 
        bounds = torch.full((rays_o.shape[0], 2), -1, dtype=torch.float32, device=rays_o.device)
        ray_aabb_intersection(rays_o, rays_d, self.bbox_center, self.bbox_size/2., bounds)
        valid = torch.all(bounds!=-1, dim=-1)

        # disable sampling under ground 
        valid = valid & self.invalid_sampling_underground(rays_o, rays_d, bounds)

        z_vals = torch.full((rays_o.shape[0], num_sample), -1, dtype=torch.float32, device=rays_o.device)
        contract_rays_o = torch.full((rays_o.shape[0], 3), -1, dtype=torch.float32, device=rays_o.device)
        contract_rays_d = torch.full((rays_o.shape[0], 3), -1, dtype=torch.float32, device=rays_o.device)

        # B(valid) x 3 
        outgoing_point = rays_o[valid] + bounds[valid,1:] * rays_d[valid]
        dir_point = outgoing_point + rays_d[valid]
        # B(valid) x 3
        contract_rays_o[valid] = self.contract_bg(outgoing_point)
        contract_rays_d[valid] = self.contract_bg(dir_point) - contract_rays_o[valid]
        # [FIXME] normalize here 
        contract_rays_d = contract_rays_d / contract_rays_d.norm(2, dim=-1, keepdim=True)

        """
        Sampling here, skip empty later 
        [FIXME]
        """

        sample_points_contract(contract_rays_o, contract_rays_d, z_vals, 
                               torch.zeros((3),dtype=torch.float32, device=self.device) - 2., 
                               torch.ones((3),dtype=torch.float32, device=self.device) * 4,
                               self.occupied_grid_bg)

        # print(torch.any(z_vals[valid] == -1).sum())
        # exit()
        # contract_bounds = torch.full((contract_rays_o.shape[0], 2), -1, dtype=torch.float32, device=rays_o.device)
        # ray_aabb_intersection(contract_rays_o, contract_rays_d, torch.zeros((3),dtype=torch.float32, device=self.device), 
        #                       torch.ones((3),dtype=torch.float32, device=self.device) * 4, contract_bounds)
        print(contract_rays_o[valid][0], contract_rays_d[valid][0])
        print(z_vals[valid][0])
        print(contract_rays_o[valid][0] + z_vals[valid][0,-1] * contract_rays_d[valid][0])
        exit()
        # contract_bounds = torch.full((contract_rays_o.shape[0], 2), -1, dtype=torch.float32, device=rays_o.device)
        # ray_aabb_intersection(contract_rays_o, contract_rays_d, torch.zeros((3),dtype=torch.float32, device=self.device), 
        #                       torch.ones((3),dtype=torch.float32, device=self.device) * 4, contract_bounds)
        # assert torch.all(contract_bounds != -1)
        # print(contract_rays_o[valid])
        # print(contract_bounds[valid,1:])
        # exit()
        # B(valid) x num_sample
        # t_vals = torch.linspace(0., 1., steps=num_sample, device=self.device)[None,:].repeat(contract_rays_o.shape[0],1)
        
        # # B(valid) x num_sample  near is always 0 
        # z_vals = t_vals * (contract_bounds[:,1:] - 1e-6)

        # [FIXME] direction ?   B(valid) x num_sample x 3  contract space 
        # 这个点直接输入Hashgrid 是 contract space 点
        # samples = contract_rays_o[:,None,:] + z_vals[...,None] * contract_rays_d[:,None.:]
        # 
        # [NOTE] 把 z_vals 转换成 
        return z_vals, valid, contract_rays_o, contract_rays_d

    """
    Rendering  Part 
    """
    def cal_integrate_weight(self, sigma, z_vals, dists, rays_d, infinity=True):
        sigma2alpha = lambda sigma, dists: 1.-torch.exp(-sigma*dists)

        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        if infinity:
            dists[:, -1] = 1e10

        if (dists < 0).sum() > 0:
            print(z_vals[torch.any(dists < 0, dim=-1)])
            print((dists < 0).sum())
            raise AssertionError
        
        alpha = sigma2alpha(sigma, dists[...,None])  # [N_rays, N_samples, 1]
        T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1, 1), device=rays_d.device), 1.-alpha + 1e-6], 1), 1)[:, :-1]
        weights = alpha * T
        return weights, T[:, -1, 0]
    
    def accumulate(self, weights, attr):
        """
        attri  B x Num_sample x channel 
        """
        return torch.sum(weights * attr, 1)

    def inference_sigma(self, samples, decoder):
        ori_shape = samples.shape
        features = self.HE(samples.reshape(-1,3))
        sigma = decoder.inference_sigma(features.reshape(*ori_shape[:-1], 32))
        return sigma 

    def compute_normal(self, samples, decoder):
        """
        samples  ... x 3 
        normal   ... x 3
        """
        samples = samples.requires_grad_(True)
        sigma = self.inference_sigma(samples, decoder)
        d_output = torch.ones_like(sigma, requires_grad=False, device=self.device)
        normal = torch.autograd.grad(
            outputs=sigma,
            inputs=samples,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        normal = -1. * normal / (normal.norm(2, dim=-1, keepdim=True) + 1e-8)

        return (normal, sigma)

    def contract_fore(self, x):
        return (x - self.min_bbox) / self.bbox_size * 4. - 2., None 
    
    def contract_bg(self, x):
        # [-2,2]
        x = (x - self.min_bbox) / self.bbox_size * 4. - 2.

        # with torch.no_grad():
        abs_x = torch.abs(x)
        L_infinity_norm, index = torch.max(abs_x, dim=-1, keepdim=True)
        # ... x 1 
        temp = 2 - 1.0 / L_infinity_norm # [1,2]

        # 2 - temp 
        # weight_feature = self.weight_bg_feature(2 - temp)

        ratio = temp / L_infinity_norm
        return x * ratio, None 

    # def contract_bg(self, x):
    #     x = (x - self.min_bbox) / self.bbox_size * 4. - 2.
    #     abs_x = torch.abs(x)
    #     L_infinity_norm, index = torch.max(abs_x, dim=-1, keepdim=True)
    #     if (L_infinity_norm == 0).sum() > 0:
    #         invalid = L_infinity_norm[...,0] == 0
    #         print(x[invalid])
    #         raise ValueError
    #     # print(L_infinity_norm.min(), L_infinity_norm.max(), abs_x.min(), abs_x.max())
    #     mask = torch.zeros(abs_x.shape, dtype=torch.bool, device=x.device)
    #     mask[torch.arange(mask.shape[0]), index.flatten()] = True
    #     return torch.where(mask, (2. - 1./(abs_x+1e-7))*torch.sign(x), x / L_infinity_norm), None 

    def render_fore_rays(self, rays_o, rays_d, num_sample, decoder, mode, occlusion_mask=None, 
                         infinity=False, **kwargs):
        
        out_dict = {}
        z_vals, dists = self.samplePoints(rays_o, rays_d, num_sample)
 
        valid = torch.all(z_vals != -1, dim = -1) 
        if occlusion_mask != None:
            valid = valid & occlusion_mask[..., 0]
        
        # if valid.sum() == 0:
        #     # print("Error!!!!!!!!!!!!!!!!!! no training rays found")
        #     return None, False

        rgb = torch.zeros_like(rays_o)
        depth = torch.zeros_like(rays_d[...,:1])
        transparency = torch.ones_like(rays_d[...,:1])
        # specular = torch.zeros_like(rays_o)
        # diffuse = torch.zeros_like(rays_o)

        # with torch.no_grad():÷
        out, ret = self.render_batch_rays(rays_o[valid], rays_d[valid], z_vals[valid], dists[valid], decoder, mode, 
                                    self.contract_fore,
                                    out_normal=False,
                                    infinity=infinity,
                                    global_step=kwargs["global_step"])
        
        if ret is False:
            return None, False
        

        # ################# [FIXME] Here ===============================
        # z_vals_mid = 0.5 * (z_vals[valid,1:] + z_vals[valid,:-1])
        # z_vals_fine = self.sample_pdf(z_vals_mid, out["weights"][...,1:-1,0], 64)
        # z_vals, _ = torch.sort(torch.cat([z_vals[valid], z_vals_fine], -1), -1)
        # dists = z_vals[:,1:] - z_vals[:,:-1]
        # dists = torch.cat([dists, 1e-6*torch.ones(dists[...,:1].shape, device=self.device)], -1)
        # out, ret = self.render_batch_rays(rays_o[valid], rays_d[valid], z_vals, dists, decoder, mode, 
        #                              self.contract_fore,
        #                              out_normal=mode==TRAIN,
        #                              infinity=infinity,
        #                              global_step=kwargs["global_step"])
        # if ret is False:
        #     return None, False
        # ################## [FIXME] Here ===============================
        
        out_dict.update(out)

        rgb[valid] = out["rgb"]
        depth[valid] = out["depth"]
        transparency[valid, 0] = out["T_left"]
        # specular[valid] = out["specular"]
        # diffuse[valid] = out["diffuse"]

        out_dict.update({"fore_valid": valid, "pred_color": rgb, "pred_depth": depth,
                          "T_left": transparency})

        return out_dict, True 
        
    def render_bg_rays(self, rays_o, rays_d, num_sample, decoder, mode,  occlusion_mask=None, 
                       infinity=True, **kwargs):

        out_dict = {}

        rgb = torch.zeros_like(rays_o)
        depth = torch.zeros_like(rays_d[...,:1])
        transparency = torch.ones_like(rays_d[...,:1])
        specular = torch.zeros_like(rays_o)
        diffuse = torch.zeros_like(rays_o)

        if kwargs["bg_mode"] == "IZ":
            z_vals, dists, valid = self.inverse_z_sampling(rays_o, rays_d, num_sample, kwargs["invalid_underground"])
            # z_vals, valid = self.sample_points_contract_space(rays_o, rays_d, num_sample)
        elif kwargs["bg_mode"] == "BS":
            z_vals, dists, valid = self.background_sampling(kwargs["fmesh"], rays_o, rays_d, num_sample)
        else:
            return None, False 

        if occlusion_mask != None:
            valid = valid & occlusion_mask[..., 0]

        # samples = rays_o[valid, None, :] + z_vals[valid, :64, None] * rays_d[valid,None,:]
        # from tools import tools 
        # tools.points2obj("/data/wxc/data/sig23/community/sample.obj", samples.detach().cpu().numpy().reshape(-1,3))
        # exit()
        
        out, ret = self.render_batch_rays(rays_o[valid], rays_d[valid], z_vals[valid], dists[valid], decoder, mode, 
                                        self.contract_bg, 
                                        out_normal=False,
                                        infinity=infinity,
                                        global_step=kwargs["global_step"])
        
        if ret == False:
            return None, ret
        # valid = torch.any(z_vals != -1, dim = -1) 
        # if occlusion_mask != None:
        #     valid = valid & occlusion_mask[..., 0]
        
        

        out_dict.update(out)

        rgb[valid] = out["rgb"]
        depth[valid] = out["depth"]
        transparency[valid, 0] = out["T_left"]
        specular[valid] = out["specular"]
        diffuse[valid] = out["diffuse"]


        out_dict.update({"valid": valid, "rgb": rgb, "depth": depth, 
                        "specular": specular, "diffuse": diffuse, 
                        "T_left": transparency})

        return out_dict, ret
    

    def render_batch_rays(self, rays_o, rays_d, z_vals, dists, decoder, mode, contract_func,  out_normal=False, 
                          infinity=False, **kwargs):
        
        # print(z_vals)

        if z_vals.shape[0] == 0:
            return None, False 
        
        
        # shape of z_vals   batch_size x num_sample 

        samples = rays_o[:, None, :] + z_vals[..., None] * rays_d[:,None,:]
        
        # print(rays_o.min(), rays_o.max(), rays_d.min(), rays_d.max())
        # print(z_vals)

        # from tools import tools 
        # tools.points2obj("./sample.obj", samples.detach().cpu().numpy().reshape(-1,3))
        # input()
        
        num_sample = samples.shape[1]

        if contract_func != None:
            contract_x, weight_feature = contract_func(samples.reshape(-1,3))
        else:
            contract_x = samples.reshape(-1,3)

        # print(contract_x)

        # print(contract_x.min(), contract_x.max())
        features = self.HE(contract_x)
        features = features.reshape(rays_o.shape[0], num_sample, 32)
        
        # 1 x 1 x 32
        step_weight_feature = self.weight_feature(kwargs["global_step"])[None,None,:].repeat_interleave(2, dim=-1)
        if weight_feature != None:
            step_weight_feature = step_weight_feature * weight_feature.reshape(rays_o.shape[0], num_sample, 32)

        inputs = torch.cat([features, rays_d[:,None,:].repeat(1,num_sample,1)], -1)
        output = decoder(inputs, 
                         weight_feature=step_weight_feature)
        # print(output["sigma"].reshape(-1,).detach().cpu().numpy())

        # temp_z_vals = z_vals.detach().cpu().numpy()[0]
        # temp_sigma = output["sigma"].reshape(-1,).detach().cpu().numpy()
        # temp_features = features[0].detach().cpu().numpy()
        # print(temp_z_vals.shape, temp_sigma.shape, temp_features.shape)
        # # m = np.concatenate([temp_z_vals[...,None], temp_sigma[...,None], temp_features], -1)
        # for a,b,c in zip(temp_z_vals, temp_sigma,temp_features):
        #     print(a,b)
        #     print(c)
        # exit()

        out = {}
        weights, T_left = self.cal_integrate_weight(output['sigma'], z_vals, dists, rays_d, infinity=infinity)
        depth = self.accumulate(weights, z_vals[..., None])
        # tint = self.accumulate(weights, output['tint'])
        # diffuse = self.accumulate(weights, output['diffuse'])
        # specular = self.accumulate(weights, output['tint'] * output['specular'])
        # out['diffuse'] = diffuse 
        # out['tint'] = tint 
        # out['specular'] = specular
        # rgb = torch.clamp(diffuse + specular, 0, 1)
        rgb = self.accumulate(weights, output["rgb"])
        out.update({"rgb":rgb,'depth': depth, "T_left": T_left, "weights": weights})


        if out_normal:
            d_output = torch.ones_like(output["sigma"], requires_grad=False, device=self.device)
            p_normal = torch.autograd.grad(
                outputs=output["sigma"],
                inputs=samples,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            p_normal = -1. * p_normal / (p_normal.norm(2, dim=-1, keepdim=True) + 1e-8)
            grad_normal = self.accumulate(weights, p_normal.detach())
            out["normal"] = grad_normal
            # out["grad_normal"] = p_normal

        # if mode is TRAIN:
        #     # [FIXME] if should weights.detach ?
        #     l2_reg_specular = torch.mean(self.accumulate(weights.detach(), (output['specular'] - 0) ** 2))
        #     out['l2_reg_specular'] = l2_reg_specular

        return out, True 