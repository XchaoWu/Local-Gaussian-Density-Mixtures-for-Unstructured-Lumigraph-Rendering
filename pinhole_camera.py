import torch.nn as nn 
import torch 
import camera 
import numpy as np 
from easydict import EasyDict as edict
from cuda import gen_rays_cuda, gen_image_rays_cuda

@torch.no_grad()
def prealign_cameras(pose,pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1,1,3,device=pose.device)
    center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
    center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
    try:
        sim3 = camera.procrustes_analysis(center_GT,center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device=pose.device))
    # align the camera poses
    center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pose[...,:3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
    return pose_aligned,sim3

@torch.no_grad()
def evaluate_camera_alignment(pose_aligned,pose_GT):
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
    R_GT,t_GT = pose_GT.split([3,1],dim=-1)
    R_error = camera.rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    error = edict(R=R_error,t=t_error)
    return error


class PinholeCamera(nn.Module):
    def __init__(self, H, W, ks, c2ws=None, noise_weight=None, gt_c2ws=None):
        super(PinholeCamera, self).__init__()

        # assert mode in ["regress", "classify"]

        # self.mode = mode 

        self.device = ks.device 

        self.H = H 
        self.W = W 

        self.num_camera = ks.shape[0]
        self.ks = ks.to(self.device)

        if c2ws != None:
            self.ori_rts = camera.pose.invert(c2ws).to(self.device)
        else:
            c2ws = torch.tensor([1,0,0,0,0,0,1,0,0,-1,0,0], dtype=torch.float32, device=self.device).reshape(1,3,4).repeat(self.num_camera,1,1)
            # c2ws = torch.eye(4)[None, :3,:4].repeat(self.num_camera, 1, 1).to(self.device)

            # c2ws[:, :, 3] = torch.rand((self.num_camera, 3), dtype=torch.float32, device=self.device)
            c2ws[:, :, 3] = torch.ones((self.num_camera, 3), dtype=torch.float32, device=self.device) * 0.5
            self.ori_rts = camera.pose.invert(c2ws)
        # print(self.ori_rts)
        # input()

        # from tools import tools 
        # import matplotlib.pyplot as plt 
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # noise_weight = 0.001

        # for item in self.ori_rts.detach().cpu().numpy():
        #     tools.plot_camera(ax, item[:,3], item[:3,:3], scale = 0.05, color="red")


        if noise_weight != None:
            noise = noise_weight * torch.randn((self.num_camera, 6), dtype=torch.float32, device=self.device)
            self.rts = camera.pose.compose([camera.lie.se3_to_SE3(noise), self.ori_rts.clone()])
        else:
            self.rts = self.ori_rts.clone()

        # for item in self.rts.detach().cpu().numpy():
        #     tools.plot_camera(ax, item[:,3], item[:3,:3], scale = 0.05, color="black")


        # ax.set_xlim([-30, 30])
        # ax.set_ylim([-30, 30])
        # ax.set_zlim([-30, 30])

        # plt.show()

        # exit()

        if gt_c2ws != None:
            self.gt_rts = camera.pose.invert(gt_c2ws).to(self.device)
        else:
            self.gt_rts = self.ori_rts.clone()

        """
        In the case of classify, [axis_angle, mu]
        In the case of regressm [axis_angle, camera center]
        """
        self.se3_refine = nn.Parameter(torch.zeros((self.num_camera, 6), dtype=torch.float32, device=self.device))
    
    def get_rts(self):
        rts_refine = camera.lie.se3_to_SE3(self.se3_refine)
        rts = camera.pose.compose([rts_refine, self.rts])
        return rts 

    def get_poses(self):
        rts_refine = camera.lie.se3_to_SE3(self.se3_refine)
        rts = camera.pose.compose([rts_refine, self.rts])
        return camera.pose.invert(rts)

    def evaluate(self):
        rts_refine = camera.lie.se3_to_SE3(self.se3_refine)
        rts = camera.pose.compose([rts_refine, self.rts])
        rts_aligned, _ = prealign_cameras(rts, self.gt_rts)
        error = evaluate_camera_alignment(rts_aligned, self.gt_rts)

        return np.rad2deg(error.R.mean().cpu()), error.t.mean()

    def get_rays(self, ray_idx=None, view_idx=None, zoom_scale=1):
        rts = self.get_rts()
        ks = self.ks.clone()
        
        if zoom_scale > 1:
            ks[:, 0, 0] *= zoom_scale
            ks[:, 1, 1] *= zoom_scale
        
        
        
        if view_idx != None:
            rays_o = torch.full((self.H*self.W, 3), 0, dtype=torch.float32, device=self.device)
            rays_d = torch.full((self.H*self.W, 3), 0, dtype=torch.float32, device=self.device)
            up_axis = torch.full((self.H*self.W, 3), 0, dtype=torch.float32, device=self.device)
            gen_image_rays_cuda(view_idx, rts, ks, rays_o, rays_d, up_axis, self.H, self.W)
        else:
            view_idx = ray_idx[0]
            ray_idx = ray_idx[1]
            rays_o = torch.full((ray_idx.shape[0], 3), 0, dtype=torch.float32, device=self.device)
            rays_d = torch.full((ray_idx.shape[0], 3), 0, dtype=torch.float32, device=self.device)
            up_axis = torch.full((ray_idx.shape[0], 3), 0, dtype=torch.float32, device=self.device)
            gen_rays_cuda(view_idx.int(), ray_idx.int(), rts, ks, rays_o, rays_d, up_axis, self.H, self.W)
        
        # print(rts)
        # print(ks)
        # print(rays_d)
        # exit()
        return rays_o, rays_d, up_axis
        

        # if view_idx != None:
        #     rts = rts[view_idx]
        #     ks = self.ks[view_idx]
        # else:
        #     ks = self.ks 
    
        # # B x HW x 3 
        # if ray_idx != None:
        #     return camera.get_center_and_ray_v3(self.H, self.W, rts, ks, ray_idx)
        # else:
        #     return camera.get_center_and_ray(self.H, self.W, rts, ks)

    def get_local_rays(self, ray_idx=None):
        return camera.get_center_and_ray_local(self.H, self.W, ray_idx=ray_idx, intr=self.ks)

