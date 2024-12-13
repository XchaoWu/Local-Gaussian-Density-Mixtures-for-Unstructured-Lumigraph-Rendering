import torch 
import numpy as np 
import os,sys,cv2 
import math 
import lpips
import time 
from tqdm import tqdm 
from load_data import load_data 
from glob import glob 
from tools import utils 
from tools import tools 
from cuda import (
    ray_cast_cuda,
    sample_points_grid,
    get_candidate_neighbor,
    new_neighbor_score_cuda,
    pixel_level_neighbor_ranking,
    pixel_level_neighbor_ranking_render,
    pixel_level_pick_up_neighbor,
    pick_up_neighbor,
    ray_aabb_intersection,
    inference_neighbor_cuda,
    accumulate_cuda,
    project_samples_cuda,
    padding_results_cuda,
    sample_points_grid_render
)
import matplotlib.pyplot as plt 
from cfg import * 
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

"""

1. give a target view point 
2. 

"""



def TIME_TEST(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        run_time = (end_time - start_time) * 1000
        print(f"[{func.__name__}] run time : {run_time:.5f} ms")
        return result
    return wrapper


def log(name, tensor):
    total_byte = tensor.nelement() * tensor.element_size() 
    total_byte = total_byte / (1000**2)
    print("%-20s\t%.4fMB\t%s" % (name, total_byte, tensor.shape))

class Renderer:
    def __init__(self, data_dir, output_name, device):
        
        self.device = device 
        
        self.num_sample = 128 
        self.num_neighbor = 8 

        self.rescale_th = -1.0
        self.full_candidate = True
        self.is_half = True
        self.inv_z = True
        

        # for rays that without intersecion with occupied grid 
        self.background = True 
        
        # control 
        self.move_scale = 0.08

        self.data_dir = data_dir
        
        self.load_data(data_dir, output_name)
        self.toGPU()

        # self.dense2sparse()
        
        self.init()

    def init(self):
        # set focal height width
        # set init camera origin & camera rotation
        self.focal = self.src_ks[0][0,0] 
        self.update_k()
        
        self.origin = [0,0,0]
        self.azimuth = 15
        self.radius = 4
        self.center = [0,0,0]
        self.inclination = 1.5
        self.update_c2w()
        
    def load_data(self, data_dir, output_name):

        data_path = os.path.join(data_dir, "logs", output_name, "checkpoint")
        # load data from disk 
        
        # data = load_data(data_dir, DATA_TYPE, factor=4, 
        #                 load_depth=False, load_point=False,
        #                 rendering_mode=FORWARD_FACING)
        # i_train = data.i_train
        # i_test = data.i_test
        # images = data.images
        # self.test_images = images[i_test]


        occupied_grid = np.load(os.path.join(data_path, "occupied_grid.npz"))
        self.occupied_grid = torch.from_numpy(occupied_grid["occupied_grid"])
        self.bbox_corner = torch.from_numpy(occupied_grid["bbox_corner"])
        self.bbox_size = torch.from_numpy(occupied_grid["bbox_size"])
        self.bbox_center = self.bbox_corner + self.bbox_size / 2.
        
        self.log2dim_resolution = [int(math.log2(item)) for item in self.occupied_grid.shape[:3]]
        log("Occupied Grid:", self.occupied_grid)
    
        # network 
        self.params = []
        weights, bias = utils.extract_MLP_para(os.path.join(data_path, "coeffiNet.pt"))
        for w,b in zip(weights, bias):
            print(w.shape, b.shape)
            self.params += [b, w.transpose(1,0).flatten()]
            # self.params += [b, w.flatten()]
        self.params = torch.cat(self.params).half()
        # self.params = torch.cat(self.params)
        log("Coeffi Net:", self.params)
        

        im_files = glob(os.path.join(data_path, "*.png"))
        self.num_view = len(im_files)

        # self.num_candidate = min(self.num_neighbor * 3, self.num_view)

        if self.full_candidate:
            # self.num_candidate = min(200, self.num_view)
            self.num_candidate = self.num_view
        else:
            self.num_candidate = min(self.num_neighbor * 3, self.num_view)
        
        
        self.src_image = []
        self.src_view = []
        
        for i in tqdm(range(self.num_view),desc="loading per-view:"):
            npz_file = np.load(os.path.join(data_path, f"{i}.npz"))
            self.src_image += [cv2.imread(os.path.join(data_path, f"{i}.png"))[...,::-1]]
            
            # H x W x (3*num_gaussian)
            view = np.concatenate([npz_file["mu_texture"], npz_file["inv_sigma_texture"],
                                   npz_file["weight_texture"]], axis=-1)
            # npz_file["valid_mask"]  TODO
            view = view.reshape(*list(view.shape[:2]), 3, -1).transpose(0,1,3,2).reshape(*list(view.shape[:2]),-1)
            
            self.src_view += [view]
        
        self.src_view = torch.from_numpy(np.stack(self.src_view, axis=0))
        self.src_image = torch.from_numpy(np.stack(self.src_image, axis=0))

        self.H = self.src_image.shape[1] 
        self.W = self.src_image.shape[2] 

        log("Src Images:", self.src_image)
        log("Src Views:", self.src_view)

        npz_file = np.load(os.path.join(data_path, "cam.npz"))
        self.src_c2ws = torch.from_numpy(npz_file["c2ws"])
        self.src_ks = torch.from_numpy(npz_file["ks"])
        self.near = float(npz_file["near"])
        self.far = float(npz_file["far"])        
        log("Src Poses:", self.src_c2ws)
        print("near", self.near, "far", self.far)

        self.test_c2ws = torch.from_numpy(npz_file["test_c2ws"])
        self.test_ks = torch.from_numpy(npz_file["test_ks"])
        log("Test Poses:", self.test_c2ws)

        self.render_c2ws = torch.from_numpy(npz_file["render_c2ws"])
        self.render_ks = torch.from_numpy(npz_file["render_ks"])
        log("Render Poses:", self.render_c2ws)


        print(f"finished loading data from {data_path}\n")
    
    def rearange_mu(self):
        for i in tqdm(range(len(self.src_c2ws)),desc="rearange mu"):
            self.c2w = self.src_c2ws[i]
            self.k = self.src_ks[i]
            rays_o, rays_d, _ = self.ray_cast()

            scale = torch.norm(rays_d, dim=-1, keepdim=True)
            rays_d = rays_d / scale
            bounds = torch.zeros_like(rays_d[...,:2])
            ray_aabb_intersection(rays_o, rays_d, self.bbox_center, self.bbox_size, bounds)
            far = torch.minimum(bounds[...,1:], self.far * scale)
            near = torch.maximum(bounds[...,0:1], self.near * scale)
            far = far.reshape(self.H, self.W, 1).cpu()
            near = near.reshape(self.H, self.W, 1).cpu()
            
            self.src_view[i, :, :, 0::3] = \
            1.0 / (1.0 / (near+1e-8) * (1.0 - self.src_view[i, :, :, 0::3]) + 1.0 / (far+1e-8) * self.src_view[i, :, :, 0::3])


        # print(torch.isinf(self.src_view).sum())

        if self.is_half:
            self.src_view = self.src_view.half().to(self.device)
        else:
            self.src_view = self.src_view.to(self.device)

        # mu = self.src_view[34, 104, 426, 0::3]
        # inv_sigma = self.src_view[34, 104, 426, 1::3]
        # weight = self.src_view[34, 104, 426, 2::3]

        # # pi = 3.1415926
        # # max_density_per_gau = weight * inv_sigma / (((2*pi) ** 0.5))

        # print(mu)
        # print(inv_sigma)
        # print(weight)
        # exit()
    

    def draw_per_ray_gaussian(self, vidx, ridx):
        # para = self.src_view[vidx].reshape(self.H*self.W, -1)[ridx]

        para = self.src_view[vidx]

        mu = para[..., 0::3].reshape(self.H*self.W,1,-1)[ridx:ridx+1]
        inv_sigma = para[..., 1::3].reshape(self.H*self.W,1,-1)[ridx:ridx+1]
        weight = para[..., 2::3].reshape(self.H*self.W,1,-1)[ridx:ridx+1]


        self.c2w = self.src_c2ws[vidx]
        self.k = self.src_ks[vidx]

        rays_o, rays_d, up_axis = self.ray_cast()
        rays_o = rays_o.reshape(-1,3)[ridx:ridx+1]
        rays_d = rays_d.reshape(-1,3)[ridx:ridx+1]

        z_vals, dists = self.sample_points_new(rays_o, rays_d, self.num_sample)
        z_vals = z_vals[..., None]


        x = z_vals * torch.norm(rays_d, dim=-1, keepdim=True)[:,None,:]


        alpha = torch.sum(weight * self.gaussian_func(x, mu, inv_sigma), dim=-1)

        visibility = torch.exp(-torch.sum(weight * self.intergrated_gaussian_func(x, mu, inv_sigma, self.near), dim=-1))
        print(alpha.shape, visibility.shape)

        alpha = alpha.detach().cpu().numpy().flatten()
        visibility = visibility.detach().cpu().numpy().flatten()
        x = x.detach().cpu().numpy().flatten()
        plt.rcParams.update({"font.size": 12})
        fig = plt.figure(figsize=(8,4))
        plt.plot(x, alpha, color="#FF6A6A", label="density", linewidth=4.0)
        plt.plot(x, visibility, color="#696969", label="visibility", linewidth=4.0)
        # plt.legend()
        plt.grid(True, linestyle='--')
        plt.savefig("Gau.eps", dpi=300)
        # plt.show()


    def dense2sparse(self):


        mu = self.src_view[..., 0::3]
        inv_sigma = self.src_view[..., 1::3]
        weight = self.src_view[..., 2::3]


        pi = 3.1415926
        max_density_per_gau = weight * inv_sigma / (((2*pi) ** 0.5))

        # sorted_idxs = torch.argsort(max_density_per_gau, descending=True, dim=-1)[...,:2]

        # valid = torch.zeros_like(max_density_per_gau).scatter_(dim=-1, index=sorted_idxs, src=torch.ones_like(max_density_per_gau)).bool()

        # mu = mu[valid].reshape(self.num_view, self.H, self.W, -1)
        # inv_sigma = inv_sigma[valid].reshape(self.num_view, self.H, self.W, -1)
        # weight = weight[valid].reshape(self.num_view, self.H, self.W, -1)

        # # self.src_view[..., 0::3] = mu
        # # self.src_view[..., 1::3] = inv_sigma
        # # self.src_view[..., 2::3] = weight

        # self.src_view = torch.stack([mu, inv_sigma, weight], dim=-1).reshape(self.num_view, self.H, self.W, -1)
        # print(self.src_view.shape)
        # # print(inv_sigma)
        # exit()

        # # print(max_density_per_gau[0,0,0])
        # # exit()
        valid = max_density_per_gau > 0.01
        # ratio = valid.sum() / valid.nelement() * 100
        # print(f"valid gaussian ratio {ratio:.2f}%")

        inv_sigma[~valid] = 0.0
        weight[~valid] = 0.0

        self.src_view[..., 1::3] = inv_sigma
        self.src_view[..., 2::3] = weight
        # print(ratio)

        # exit()



    def toGPU(self):
        self.occupied_grid = self.occupied_grid.to(self.device)
        self.bbox_corner = self.bbox_corner.to(self.device)
        self.bbox_center = self.bbox_center.to(self.device)
        self.bbox_size = self.bbox_size.to(self.device)
        
        self.params = self.params.to(self.device)
        
        self.src_image = self.src_image.to(self.device)
        
        self.src_c2ws = self.src_c2ws.to(self.device)

        self.src_rts = self.toW2C(self.src_c2ws)
        self.src_ks = self.src_ks.to(self.device)
        
        self.proj_mat = torch.matmul(self.src_ks, self.src_rts)


        self.rearange_mu()
        
        print("finished uploading data to GPU\n")

###################### MODE ######################
    def inference_test(self):
        self.test_c2ws = self.test_c2ws.to(self.device)
        self.test_ks = self.test_ks.to(self.device)
        lpips_model = lpips.LPIPS(net='vgg')
        # psnrs = []
        # ssims = []
        # lpipss = []
        for i in tqdm(range(self.test_c2ws.shape[0]), desc="rendering"):
            self.k = self.test_ks[i]
            self.c2w = self.test_c2ws[i]
            self.render()

            # color = self.rgb_frame.detach().cpu().numpy().reshape(self.H,self.W,3) / 255.


            # gt = self.test_images[i] 
            # test_psnr = peak_signal_noise_ratio(color, gt, data_range=1.0)
            # test_ssim = structural_similarity(color, gt, win_size=11, multichannel=True, gaussian_weights=True)

            # # cv2.imwrite(f"pred-{i}-new-{test_psnr:.2f}-{test_ssim:.3f}.png", color[..., ::-1] * 255)

            # gt_lpips = torch.from_numpy(gt).to(torch.float32) * 2.0 - 1.0
            # predict_image_lpips = torch.from_numpy(color).to(torch.float32) * 2.0 - 1.0
            # lpips_result = lpips_model.forward(predict_image_lpips.permute(2,0,1), gt_lpips.permute(2,0,1)).cpu().detach().numpy()
            # test_lpips = np.squeeze(lpips_result)

            # psnrs += [test_psnr]
            # ssims += [test_ssim]
            # lpipss += [test_lpips]

        #     print(f"PSNR {test_psnr:.2f} SSIM {test_ssim:.3f} LPIPS {test_lpips:.3f}")

        # print(f"Mean PSNR {np.mean(psnrs):.2f} SSIM {np.mean(ssims):.3f} LPIPS {np.mean(lpipss):.3f}")

    def inference_render(self, data_dir):

        out_dir = os.path.join(data_dir, "output")
        if os.path.exists(out_dir) is False:
            os.mkdir(out_dir)

        self.render_c2ws = self.render_c2ws.to(self.device)
        self.render_ks = self.render_ks.to(self.device)

        c2ws = self.render_c2ws
        

        # R0 = c2ws[70].clone()
        # R0[:,3] -= R0[:,2] * 5
        # R0 = torch.from_numpy(tools.angle2rotation((0,15,0))).to(self.device).float() @ R0

        # R1 = c2ws[60].clone()
        # R1[:,3] += R1[:,2] * 0.9
        # R1[:,3] -= R1[:,0] * 1.0
        # R1 = torch.from_numpy(tools.angle2rotation((0,-10,0))).to(self.device).float() @ R1


        # # R2 = c2ws[0].clone()
        # # R2[:,3] -= R2[:,2] * 6
        # # R2[:,3] -= R2[:,0] * 0.5
        # # R2 = torch.from_numpy(tools.angle2rotation((0,-15,0))).to(self.device).float() @ R2
        
        # c2ws = torch.stack([R0, R1], dim=0)

        # # draw cameras 
        # # 创建3D绘图
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # 绘制相机
        # for item in c2ws.detach().cpu().numpy():
        #     tools.plot_camera(ax, item[:,3], item[:3,:3], scale = 0.5, color="#4169E1")

        # for item in self.src_c2ws.detach().cpu().numpy():
        #     tools.plot_camera(ax, item[:,3], item[:3,:3], scale = 0.5, color="black")

        # # 设置坐标轴范围
        # ax.set_xlim([-10, 10])
        # ax.set_ylim([-10, 10])
        # ax.set_zlim([-10, 10])

        # plt.show()



        for i in tqdm(range(c2ws.shape[0]), desc="rendering"):
            self.k = self.render_ks[i]
            self.c2w = c2ws[i]
            self.render()

            cv2.imwrite(os.path.join(out_dir, f"{i}.png"), 
                        self.rgb_frame.cpu().numpy().reshape(self.H, self.W, 3)[...,::-1])
            # depth = self.depth_frame.cpu().numpy().reshape(self.H, self.W, 1).repeat(3, -1)
            # depth = (depth - self.near) / (self.far - self.near) * 255
            # cv2.imwrite(os.path.join(out_dir, f"d-{i}.png"), depth)    

    def inference_path(self, data_dir):
        ks, c2ws = tools.read_campara(os.path.join(data_dir, "render_ours.log"))

        ks = torch.from_numpy(ks).to(self.device)
        c2ws = torch.from_numpy(c2ws).to(self.device)

        # ks[:, 0, 0] *= 2
        # ks[:, 1, 1] *= 2

        out_dir = os.path.join(data_dir, "output")
        if os.path.exists(out_dir) is False:
            os.mkdir(out_dir)

        # R0 = c2ws[70].clone()
        # R0[:,3] -= R0[:,2] * 3
        # R0 = torch.from_numpy(tools.angle2rotation((0,15,0))).to(self.device).float() @ R0

        # R1 = c2ws[60].clone()
        # R1[:,3] += R1[:,2] * 1.5
        # R1 = torch.from_numpy(tools.angle2rotation((0,0,0))).to(self.device).float() @ R1


        # # R2 = c2ws[0].clone()
        # # R2[:,3] -= R2[:,2] * 6
        # # R2[:,3] -= R2[:,0] * 0.5
        # # R2 = torch.from_numpy(tools.angle2rotation((0,-15,0))).to(self.device).float() @ R2
        
        # c2ws = torch.stack([R0, R1], dim=0)

        # # draw cameras 
        # # 创建3D绘图
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # 绘制相机
        # for item in c2ws.detach().cpu().numpy():
        #     tools.plot_camera(ax, item[:,3], item[:3,:3], scale = 0.5, color="#4169E1")

        # for item in self.src_c2ws.detach().cpu().numpy():
        #     tools.plot_camera(ax, item[:,3], item[:3,:3], scale = 0.5, color="black")

        # # 设置坐标轴范围
        # ax.set_xlim([-10, 10])
        # ax.set_ylim([-10, 10])
        # ax.set_zlim([-10, 10])

        # plt.show()

        for i in tqdm(range(c2ws.shape[0]), desc="rendering"):

            # i = 119

            self.k = ks[i]
            self.c2w = c2ws[i]

            # R = torch.from_numpy(tools.angle2rotation((0,15,0))).to(self.device).float()
            # self.c2w = R @ self.c2w

            self.render()

            cv2.imwrite(os.path.join(out_dir, f"{i}.png"), 
                        self.rgb_frame.cpu().numpy().reshape(self.H, self.W, 3)[...,::-1])
            # exit()
            depth = self.depth_frame.cpu().numpy().reshape(self.H, self.W, 1).repeat(3, -1)
            depth = (depth - self.near) / (self.far - self.near) * 255
            cv2.imwrite(os.path.join(out_dir, f"d-{i}.png"), depth)     
            # rgb = self.rgb_frame.cpu().numpy().reshape(self.H, self.W, 3)[...,::-1]
            # out = np.concatenate([rgb, depth], 1)
            
            # cv2.imwrite(os.path.join(out_dir, f"{i}.png"), out)       

            # exit()

###################### OPERATION ##################

    def move_forward(self):
        self.origin  = self.origin + self.c2w[:, 2] * self.move_scale
        self.update_c2w()

    def move_back(self):
        self.origin  = self.origin - self.c2w[:, 2] * self.move_scale
        self.update_c2w()
        
    def move_left(self):
        self.origin = self.origin - self.c2w[:,0] * self.move_scale
        self.update_c2w()
        
    def move_right(self):
        self.origin = self.origin + self.c2w[:,0] * self.move_scale
        self.update_c2w()
        
    def move_up(self):
        self.origin = self.origin - self.c2w[:,1] * self.move_scale
        self.update_c2w()
        
    def move_down(self):
        self.origin = self.origin + self.c2w[:,1] * self.move_scale
        self.update_c2w()
        
    def zoom(self, scale):
        self.focal *= scale
        self.update_k()
        
    def rotate(self, delta_x, delta_y):
        self.azimuth += (delta_x / 2)
        self.inclination += (delta_y / 2)
        eps = 0.001
        self.inclination = min(max(eps, self.inclination), math.pi - eps)
        self.update_c2w()

    def lookat(self, look_from, look_to, tmp = np.asarray([0., -1., 0.])):
        forward = look_from - look_to
        forward = forward / np.linalg.norm(forward)
        right = np.cross(tmp, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        c2w = np.zeros((3,4))
        c2w[:,0] = right
        c2w[:,1] = up 
        c2w[:,2] = forward
        c2w[:,3] = np.array(self.origin).astype(np.float32)
        return c2w 
    
    def update_k(self):
        self.k = torch.tensor([self.focal, 0, self.W/2.,
                               0, self.focal, self.H/2.,
                               0,0,1], dtype=torch.float32, device=self.device)
    
    def update_c2w(self):
        offset = np.asarray([self.radius * math.cos(-self.azimuth) * math.sin(self.inclination),
                             self.radius * math.cos(self.inclination),
                             self.radius * math.sin(-self.azimuth) * math.sin(self.inclination)])
        look_from = self.center
        look_to = self.center + offset
        self.c2w = torch.tensor(self.lookat(look_from, look_to), dtype=torch.float, device=self.device)
    
    def toW2C(self, c2ws):
        rs = c2ws[:, :3,:3].permute(0,2,1)
        cs = c2ws[:, :3,3:4]
        ts = - rs @ cs
    
        return torch.cat([rs, ts], dim=-1)

##################### BASIC FUNCTION  ###############################
    
    # @TIME_TEST
    def ray_cast(self):

        rays_o = torch.full((self.H*self.W, 3), 0, dtype=torch.float32, device=self.device)
        rays_d = torch.full((self.H*self.W, 3), 0, dtype=torch.float32, device=self.device)
        up_axis = torch.full((self.H*self.W, 3), 0, dtype=torch.float32, device=self.device)
        ray_cast_cuda(self.toW2C(self.c2w[None,...])[0], self.k, rays_o, rays_d, up_axis, self.H, self.W)
        return rays_o, rays_d, up_axis

    @torch.no_grad()
    def sample_points(self, rays_o, rays_d, num_sample, sample_mode, inv=False):
        
        if sample_mode == GRID:
            return self.sparse_grid_sampling(rays_o, rays_d, num_sample)
        
        elif sample_mode == NEAR_FAR:
            return self.near_far_grid_sampling(rays_o, rays_d, num_sample, inv)
        
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
        print(print(z_vals))
        print("near", near, "far", far)
        if inv:
            z_vals = 1./(1./near * (1.-z_vals[None,:]) + 1./ far * (z_vals[None,:]))
        else:
            z_vals = z_vals[None,:] * (far - near) + near  
        print(z_vals)
        exit()
        
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
                        self.occupied_grid, self.near, self.far, *self.log2dim_resolution)
        
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        
        return z_vals, dists, bounds


    @TIME_TEST
    def sample_points_new(self, rays_o, rays_d, num_sample):
        z_vals = torch.full((rays_o.shape[0], num_sample), -1, dtype=torch.float32, device=self.device)
        dists = torch.full((rays_o.shape[0], num_sample), 0.0, dtype=torch.float32, device=self.device)
        sample_points_grid_render(rays_o, rays_d, z_vals, dists, self.bbox_corner, self.bbox_size,
                        self.occupied_grid, self.near, self.far, *self.log2dim_resolution, 
                        self.inv_z, self.background)
        z_vals, _  = z_vals.sort(dim=-1)

        # print(z_vals)
        # exit()

        dists = torch.cat([z_vals[...,1:] - z_vals[...,:-1], 1e10*torch.ones(rays_o.shape[0], 1, device=self.device)], dim=-1)
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        
        return z_vals, dists
    

    @TIME_TEST
    def sample_points_LLFF(self, rays_o, rays_d, num_sample):
        
        # return self.near_far_sampling(rays_o, rays_d, num_sample, inv=True, near=self.near, far=self.far)
        coarse_num = 64
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
    
    # @TIME_TEST
    # def sample_points(self, rays_o, rays_d):

    #     z_vals = torch.full((rays_o.shape[0], self.num_sample), -1, dtype=torch.float32, device=self.device)
    #     dists = torch.full((rays_o.shape[0], self.num_sample), 0.0, dtype=torch.float32, device=self.device)
    #     sample_points_grid(rays_o, rays_d, z_vals, dists, self.bbox_corner, self.bbox_size,
    #                     self.occupied_grid, self.near, self.far, *self.log2dim_resolution)
        
    #     dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
    #     return z_vals, dists

    @TIME_TEST
    def get_candidate_neighbor(self, rays_o, ref_idxs):
        
        distance = torch.norm(self.c2w[:,3] - self.src_c2ws[:,:,3], dim=-1)
        
        sorted_distance, sorted_idxs = torch.sort(distance, dim=-1, descending=False)
        
        candidate_neighbors = sorted_idxs[:self.num_candidate]

        candidate_neighbors = candidate_neighbors[None,:].repeat(rays_o.shape[0], 1)
        
        return candidate_neighbors.int()
    
    @TIME_TEST
    def neighbor_view_selection(self, rays_o, rays_d, up_axis, z_vals):
        
        B, N = z_vals.shape[:2]
        
        ref_idxs = torch.ones_like(rays_o[...,:1]).int() * -1
        
        candidate_neighbors = self.get_candidate_neighbor(rays_o, ref_idxs)


        # print(candidate_neighbors[0].detach().cpu().numpy().tolist())

        # temp_score = torch.full((B, N, self.num_candidate, 4), 0, dtype=torch.float32, device=self.device)
        # # rewrite 
        # new_neighbor_score_cuda(ref_idxs, up_axis, rays_o, rays_d, samples,
        #                         self.toW2C(self.src_c2ws), self.src_ks, candidate_neighbors, temp_score, 
        #                         0, self.H, self.W)
        
        # temp_score, temp_nei_idxs = torch.sort(temp_score, dim=2, descending=True)
        # temp_nei_idxs[temp_score <= 0] = -1
        # # return nei_idxs
        score = torch.full((B, self.num_candidate, 4), 0.0, dtype=torch.float32, device=self.device)
        # print(self.proj_mat.shape)
        # print(self.proj_mat[13])
        # exit()
        # print(candidate_neighbors.shape)
        # exit()
        # s = time.time()
        pixel_level_neighbor_ranking_render(candidate_neighbors, self.proj_mat, self.src_c2ws[:,:,3],
                                            rays_o, rays_d, z_vals, up_axis, score, 512, self.H, self.W)
        # torch.cuda.synchronize()
        # e = time.time()
        # print(e-s)

        # print(score[self.idx])
        # exit()
        # print(score[-1])
        # # print(score.flatten()[51819260])
        # exit()

        # B x num_candidate x 4
        # s = time.time()
        sorted_score, sorted_idxs = torch.sort(score, dim=1, descending=True)
        # torch.cuda.synchronize()
        # e = time.time()
        # print(e-s)
        sorted_idxs[sorted_score <= 0] = -1
        sorted_idxs = sorted_idxs.int()

        # print( (sorted_idxs[self.idx] != -1 ).sum() )
        # exit()

        nei_idxs = torch.full((B, self.num_candidate, 1), -1, dtype=torch.int32, device=self.device)
        # s = time.time()
        pixel_level_pick_up_neighbor(sorted_idxs.int(), candidate_neighbors, nei_idxs)
        # torch.cuda.synchronize()
        # e = time.time()
        # print(e-s)

        # print(nei_idxs[self.idx].flatten())
        # exit()
        # print(nei_idxs[-1])
        # s = time.time()
        nei_idxs = nei_idxs[:,:self.num_neighbor]
        # print(nei_idxs.shape)
        # nei_idxs = nei_idxs.repeat_interleave(step, dim=1)
        # torch.cuda.synchronize()
        # e = time.time()
        # print(e-s)
        # print(nei_idxs[-1])
        # exit()
        
        return nei_idxs 

    def cal_weight(self, alpha):
        T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1, 1), device=self.device), 1.-alpha], 1), 1)[:, :-1]
        weight = T * alpha
        
        # sum_weight = torch.sum(weight, dim=1, keepdim=True) + 1e-8
        # sum_weight[sum_weight > 0.95] = 1.0   
        # weight = weight / sum_weight
        
        return weight 
    
    @TIME_TEST
    def inference(self, rays_o, rays_d, z_vals, dists, nei_idxs):
        
        B = rays_d.shape[0]
        
        # num_step = self.num_sample
        num_step = 16
        step_num = self.num_sample // num_step

        T = torch.full((B,), 1.0, dtype=torch.float32, device=self.device)
        depth = torch.full((B,1), 0.0, dtype=torch.float32, device=self.device)
        color = torch.full((B,3), 0.0, dtype=torch.float32, device=self.device)


        # nei_dir = torch.full((B, step_num, self.num_neighbor, 3), 0, dtype=torch.float32, device=self.device)
        # warped_uvz = torch.full((B, step_num, self.num_neighbor, 3), -1, dtype=torch.float32, device=self.device)
        count = torch.full((B,1), 0, dtype=torch.int32, device=self.device)


        # samples = rays_o[:,None,:] + z_vals[:,:, None] * rays_d[:,None,:]
        # from tools import tools 
        # tools.points2obj(os.path.join(self.data_dir, "samples.obj"), samples.detach().cpu().numpy().reshape(-1,3))
        # exit()

        for i in range(num_step):
        
            blend_alpha = torch.full((B, step_num, 1), 0, dtype=torch.float32, device=self.device)

            blend_color = torch.full((B, step_num, 3), 0, dtype=torch.float32, device=self.device)

            mask = torch.full((B, step_num, 1), 0, dtype=torch.bool, device=self.device)
            
            samples = rays_o[:,None,:] + z_vals[:,i*step_num:(i+1)*step_num, None] * rays_d[:,None,:]

            # s = time.time()
            # project_samples_cuda(self.c2w[:,3], samples, nei_idxs, self.proj_mat,
            #                      self.src_c2ws[:,:,3], self.params, coeffi,
            #                      self.H, self.W)
            # torch.cuda.synchronize()
            # e = time.time()
            # print(e-s)

            # if i == num_step - 1:
                # from tools import tools 
                # tools.points2obj(os.path.join(self.data_dir, "samples.obj"), samples.detach().cpu().numpy().reshape(-1,3))
                # exit()

            # print(self.src_view.shape)
            inference_neighbor_cuda(self.c2w[:,3], samples, self.src_view, self.src_image, 
                                    nei_idxs, self.proj_mat, self.src_c2ws[:,:,3],
                                    self.params, self.bbox_center, self.bbox_size,
                                    blend_alpha, blend_color, mask, self.near, self.far, self.is_half)
            
            # input()
            # # print(alpha.min(), alpha.max(), visibility.min(), visibility.max())
            # # exit()
            # # blend_alpha = torch.sum(alpha * visibility, dim=2) / (torch.sum(visibility, dim=2)+EPS)
            # print(blend_alpha)
            # # exit()
            count += mask.sum(dim=1)
            # print(blend_alpha.flatten())
            # print(blend_alpha[self.idx])
            blend_alpha = 1. - torch.exp(-blend_alpha * dists[:, i*step_num:(i+1)*step_num, None])
            # print(torch.cat([blend_color,blend_alpha], dim=-1))
            # input()

            # print(nei_idxs.flatten())
            # print(blend_color.flatten())
            # print(blend_alpha.flatten())
            # print(T)
            # print(depth, self.near, self.far)
            # exit()

            # print(blend_color[self.idx].reshape(-1,3))
            # print(blend_alpha[self.idx])
            # print(T[self.idx])
            # print(depth[self.idx], self.near, self.far)
            # input()

            weight = torch.full((B, step_num, 1), 0, dtype=torch.float32, device=self.device)

            accumulate_cuda(blend_alpha, T, weight)

            depth = depth + torch.sum(weight * z_vals[:,i*step_num:(i+1)*step_num, None], dim=1)

            color = color + torch.sum(weight * blend_color, dim=1)
        
        # valid = count > self.num_sample / 2. 

        # padding_results_cuda(valid.reshape(self.H,self.W,1), color.reshape(self.H,self.W,3))

        # mask = T > 0.05
        # color[mask] /= (1 - T[mask])[..., None]
        # print(depth)
        # exit()
            
        return color, depth

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

    def cal_ref_weight(self, alpha, visibility):
        weight = (visibility[:, :-1] - visibility[:, 1:]).clamp(0,1)
        weight = torch.cat([weight, 1. - torch.sum(weight, dim=1, keepdim=True).clamp(0,1)], dim=1)
        # weight = torch.cat([weight, visibility[:,-1:]], dim=1)
        # weight = weight / (torch.sum(weight, dim=1, keepdim=True)+1e-8)
        return weight
    
    def inference_per_view(self, vidx):
        """
        如果per-view是清晰的，那么是投影的问题
        """
        para = self.src_view[vidx]

        mu = para[..., 0::3].reshape(self.H*self.W,1,-1)
        inv_sigma = para[..., 1::3].reshape(self.H*self.W,1,-1)
        weight = para[..., 2::3].reshape(self.H*self.W,1,-1)

        # idx = 104 * self.W + 426
        # print(mu[idx:idx+1].flatten())
        # print(inv_sigma[idx:idx+1].flatten())
        # print(weight[idx:idx+1].flatten())

        self.c2w = self.src_c2ws[vidx]
        self.k = self.src_ks[vidx]

        rays_o, rays_d, up_axis = self.ray_cast()

        z_vals, dists = self.sample_points_LLFF(rays_o, rays_d, self.num_sample)
        z_vals = z_vals[..., None]

        # print(z_vals[idx:idx+1].flatten())

        x = z_vals * torch.norm(rays_d, dim=-1, keepdim=True)[:,None,:]
        # print(x[idx:idx+1].flatten())

        # exit()
        # x =  rays_o[:,None,:] + z_vals[..., None] * rays_d[:,None,:]

        alpha = torch.sum(weight * self.gaussian_func(x, mu, inv_sigma), dim=-1)

        visibility = torch.exp(-torch.sum(weight * self.intergrated_gaussian_func(x, mu, inv_sigma, self.near), dim=-1))
        print(alpha.shape, visibility.shape)

        weight = self.cal_ref_weight(alpha, visibility)
        print(weight.shape, z_vals.shape)
        depth = torch.sum(weight[..., None] * z_vals, dim=1)
        
        depth = depth.detach().cpu().numpy().reshape(self.H, self.W, 1)

        depth = (depth - self.near) / (self.far - self.near) * 255
        # cv2.imwrite("depth.png", depth)
        # exit()
        # cv2.imwrite("color.png", color.cpu().numpy().reshape(self.H, self.W, 3).astype(np.uint8)[...,::-1])
        cv2.imshow("frame", depth.reshape(self.H, self.W, 1).astype(np.uint8))
        cv2.waitKey(0)
        
            
    @TIME_TEST
    def render(self, **kwargs):
        
        """
        0. compute ray 
        1. Ray marching based sample point along the ray 
        2. View selection for each point (time ? )
        3. Inference Visibility Alpha , get warped color 
        4. blending 
        """
        
        # (HW) x 3
        rays_o, rays_d, up_axis = self.ray_cast()

        # px = 405
        # py = 273
        # idx = py * self.W + px

        # # idx = 0
        # self.idx = idx

        # rays_o = rays_o[idx:idx+1]
        # rays_d = rays_d[idx:idx+1]
        # self.idx = 0
        
        # (HW) x num_sample  only spase grid sampling
        # z_vals, dists = self.sample_points(rays_o, rays_d)

        # z_vals_old, dists = self.sample_points_LLFF(rays_o, rays_d, self.num_sample)

        z_vals, dists = self.sample_points_new(rays_o, rays_d, self.num_sample)
        # print(z_vals_old[0].flatten())
        # print(z_vals[0].flatten())
        # # print((z_vals_old - z_vals).abs().sum())
        # input()
        # print(z_vals_old[0].flatten())
        # print(z_vals[0].flatten())
        # exit()
        
        # print(dists[idx])
        # input()
        # print(rays_o[:,None,:].reshape(-1,3))
        # # print(z_vals.flatten())
        # samples = rays_o[:,None,:] + z_vals[:,:, None] * rays_d[:,None,:]
        # # print(samples.reshape(-1,3))
        # print(rays_o)
        # print(rays_d)
        # x = samples.reshape(-1,3)
        # print(z_vals)
        # print("nei o",self.src_c2ws[34:35,:3,3])
        # item1 = (x - self.src_c2ws[34:35,:3,3])
        # print(item1)
        # x_cam = torch.sum(self.src_c2ws[34,:3,:3].transpose(1,0)[None,:,:] * item1[:,None,:], dim=-1) 

        # print(x_cam)
        # exit()
        # x_ = self.src_ks[34] @ torch.sum(self.src_c2ws[34,:3,:3].transpose(1,0) * (x - self.src_c2ws[34:35,:3,3]), dim=-1) 
        # x = torch.cat([x, torch.ones_like(x[:,:1])], -1)
        # x_ = torch.sum(self.proj_mat[34, None, :, :] * x[:, None, :], dim=-1)

        # x_ = x_[..., :2] / x_[..., 2:3]
        # # print(x)
        # print(x_)
        # print(x_.shape)

        # # x_ = x_.detach().cpu().numpy()
        # img = self.src_image[34].detach().cpu().numpy()
        # for item in x_:
        #     u,v = item 
        #     img[int(v), int(u)] = (0,0,255)

        # cv2.imwrite("color.png", img[..., ::-1])

        # exit()
    
        # from tools import tools 
        # tools.points2obj(os.path.join(data_dir, "samples.obj"), samples.detach().cpu().numpy().reshape(-1,3))
        # # input()
        # # print(z_vals)
        # exit()

        nei_idxs = self.neighbor_view_selection(rays_o, rays_d, up_axis, z_vals)

        color, depth = self.inference(rays_o, rays_d, z_vals, dists, nei_idxs)
        # print(color, depth)
        # print(color[self.idx], depth[self.idx])
        # exit()
        # print(color[idx], depth[idx])
        # exit()
        # # import matplotlib.pyplot as plt 
        # plt.imshow(color)
        # cv2.imwrite("color.png", color.cpu().numpy().reshape(self.H, self.W, 3).astype(np.uint8)[...,::-1])
        # cv2.imshow("frame", color.cpu().numpy().reshape(self.H, self.W, 3).astype(np.uint8)[...,::-1])
        # depth = depth.detach().cpu().numpy()
        # depth = (depth - self.near) / (self.far - self.near) * 255
        # depth = depth.reshape(self.H, self.W, 1)
        # cv2.imshow("depth", depth.astype(np.uint8))
        # cv2.waitKey(0)
        # print(torch.where( (color[..., 0] < 10) & (color[..., 1] < 10) & (color[..., 2] < 10)))
        # exit()

        self.rgb_frame = color
        self.depth_frame = depth


        # color = color.detach().cpu().numpy().reshape(self.H,self.W,3) / 255.
        # gt = cv2.imread("0GT.png")[...,::-1] / 255.
        # pred = cv2.imread("color0.png")[...,::-1] / 255.
        # # print(color.min(), color.max())

        # test_psnr = peak_signal_noise_ratio(pred, gt, data_range=1.0)
        # test_ssim = structural_similarity(pred, gt, win_size=11, multichannel=True, gaussian_weights=True)
        # print("origin")
        # print("test_psnr", test_psnr)
        # print("test_ssim", test_ssim)

        # test_psnr = peak_signal_noise_ratio(color, gt, data_range=1.0)
        # test_ssim = structural_similarity(color, gt, win_size=11, multichannel=True, gaussian_weights=True)
        # print("now")
        # print("test_psnr", test_psnr)
        # print("test_ssim", test_ssim)

        # cv2.imwrite("color.png", color[...,::-1] * 255)
        # depth = depth.detach().cpu().numpy()
        # depth = (depth - self.near) / (self.far - self.near) * 255
        # depth = depth.reshape(self.H, self.W, 1)
        # cv2.imwrite("depth.png", depth)


        # exit()


if __name__ == "__main__":


    
    gpuIdx = int(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuIdx}"
    device = torch.device("cuda:0")
    

    # DATA_TYPE = SELFDATA
    # data_dir = "/data/wxc/sig24/data/data/undistort_llff/fern_undistort/logs/CLOUD--d4-mlp-2024-04-29-12-25/checkpoint/"
    # data_dir = "/data/wxc/sig24/data/data/self_data/red_car/logs/CLOUD--NEW-VERSION-2024-05-01-16-33/checkpoint"
    # data_dir = "/data/wxc/sig24/data/data/self_data/red_car/logs/CLOUD--base-2024-04-29-20-01/checkpoint"
    # data_dir = "/data/wxc/sig24/data/data/undistort_llff/fern_undistort"
    # data_dir = "/data/wxc/sig24/data/data/undistort_llff/trex_undistort"


    # data_dir = "/data/wxc/sig24/data/data/self_data/natatorium"
    # output_name = "CLOUD--no-perview-inv-z-2024-05-06-04-49"

    # data_dir = "/data/wxc/sig24/data/data/self_data/skyscraper"
    # output_name = "CLOUD--BASE-60k-2024-05-09-22-19"


    # data_dir = "/data/wxc/sig24/data/data/self_data/mall"
    # output_name = "CLOUD--BASE-60k-2024-05-09-21-58"

    # data_dir = "/data/wxc/sig24/data/data/self_data/bicycle"
    # output_name = "CLOUD--BASE-60k-2024-05-09-22-33"

    # data_dir = "/data/wxc/sig24/data/data/self_data/red_car"
    # output_name = "CLOUD--no-perview-inv-z-2024-05-05-22-54"

    # data_dir = "/data/wxc/sig24/data/data/self_data/portrait_dense"
    # output_name = "CLOUD--BASE-60k-2024-05-05-11-39"
    # output_name = "CLOUD--no-voting-2024-05-07-04-23"
    # output_name = "CLOUD--S-2-2024-05-07-19-46"
    # output_name = "CLOUD--S-4-2024-05-08-00-06"
    # output_name = "CLOUD--S-8-2024-05-08-03-59"


    # data_dir = "/data/wxc/sig24/data/data/self_data/blue_car_blur"
    # output_name = "CLOUD--TEST-2024-07-05-22-31"


    # data_dir = "/data/wxc/sig24/data/data/self_data/blue_car_dense"
    # output_name = "CLOUD--BASE-60k-2024-05-11-16-03"
    # output_name = "CLOUD--exp-weight-2024-05-12-17-29"

    # data_dir = "/data/wxc/sig24/data/data/self_data/bull"
    # output_name = "CLOUD--BASE-60k-2024-05-12-17-00"

    # data_dir = "/data/wxc/sig24/data/data/self_data/bull_v2"
    # output_name = "CLOUD--BASE-60k-2024-05-12-20-58"

    # data_dir = '/data/wxc/sig24/data/data/NPC_data/compost/'
    # output_name = "CLOUD--NEW-BASE-2024-05-14-11-45"


    # data_dir = "/data/wxc/sig24/data/data/self_data/bull_debug1"
    # output_name = "CLOUD--BASE-60k-2024-05-16-12-41"

    # data_dir = "/data/wxc/sig24/data/data/self_data/bull_debug5"
    # output_name = "CLOUD--BASE-60k-2024-05-17-21-42"

    # data_dir = "/data/wxc/sig24/data/data/undistort_llff/fern_undistort/"
    # output_name = "CLOUD--BASE-60k-2024-05-02-11-48"


    # data_dir = "/data/wxc/sig24/data/data/undistort_llff/trex_undistort/"
    # output_name = "CLOUD--base-60k-2024-05-02-11-49"


    # data_dir = "/data/wxc/sig24/data/data/undistort_llff/food_undistort/"
    # output_name = "CLOUD--BASE-60k-2024-05-03-22-55"

    # data_dir = "/data/wxc/sig24/data/data/undistort_llff/seasoning_undistort/"
    # output_name = "CLOUD--NEW-BASE-2024-05-07-08-14"


    # data_dir = "/data/wxc/sig24/data/data/self_data/sculpture_dense"
    # output_name = "CLOUD--BASE-60k-2024-05-14-21-15"

    # data_dir = "/data/wxc/sig24/data/data/self_data/blue_car_dense"
    # output_name = "CLOUD--NOISE-0.001-2024-08-07-16-31"

    data_dir = "/data/wxc/sig24/data/data/undistort_llff/cd_undistort"
    output_name = "CLOUD--new-plane-2024-08-09-16-21"
    
    renderer = Renderer(data_dir, output_name, device)
    
    # renderer.inference_test()

    # renderer.inference_inter()

    # renderer.draw_per_ray_gaussian(vidx=6, ridx=1000)
    # renderer.draw_per_ray_gaussian(vidx=7, ridx=2000)
    # renderer.draw_per_ray_gaussian(vidx=3, ridx=3000)
    # renderer.draw_per_ray_gaussian(vidx=2, ridx=4000)
    # renderer.draw_per_ray_gaussian(vidx=9, ridx=5000)
    # renderer.inference_path(data_dir)
    renderer.inference_render(data_dir)
    # renderer.inference_per_view(vidx=34)
    # renderer.inference_per_view(vidx=48)
    # renderer.inference_per_view(vidx=40)
    # renderer.inference_per_view(vidx=37)
    # renderer.inference_per_view(vidx=49)
    # renderer.inference_per_view(vidx=41)
    # renderer.inference_per_view(vidx=36)
    # renderer.inference_per_view(vidx=50)