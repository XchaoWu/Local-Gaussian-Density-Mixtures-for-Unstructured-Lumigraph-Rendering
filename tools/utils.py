import numpy as np 
from tqdm import tqdm 
import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import imageio
import cv2,os,sys 
import imageio
from glob import glob 
import random 
import yaml 
import time
from easydict import EasyDict as edict
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
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


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def gen_patch_ray(H, W, num_view, patch_size, sample_num, device):
    
    num_patch = int(sample_num // (patch_size ** 2))
    assert num_patch >= 1
    
    vidx = random.randint(0, num_view-1)
    
    start_idx = np.random.choice((H-patch_size)*(W-patch_size), size=(num_patch,), p=None)
    start_idx = torch.from_numpy(start_idx)
    start_idx = (start_idx // (W - patch_size) * W) + (start_idx % (W - patch_size))

    # patch_size x patch_size
    M = torch.arange(patch_size)[None,:].repeat(patch_size, 1)
    # num_view x num_patch x patch_size x patch_size
    M = (M + M.transpose(1,0) * W)[None, None,...] + start_idx[..., None, None]
    M = M.flatten().to(device)
    
    return torch.ones_like(M) * vidx, M



# def sample_rays_with_distribution(H, W, probs, sample_num, num_view, device):

#     hard_ratio = 0.2
#     hard_num = int(sample_num * hard_ratio)
#     uniform_num = sample_num - hard_num
#     probs = probs.reshape(probs.shape[0],-1)
#     # num_view x hard_num
#     hard_idxs = probs.multinomial(num_samples=hard_num)

#     uniform_idxs = []
#     for i in range(num_view):
#         uniform_idxs += [torch.randperm(H*W, device=device)[:uniform_num]]
#     # num_view x uniform_num
#     uniform_idxs = torch.stack(uniform_idxs)
    
#     idxs = torch.cat([uniform_idxs, hard_idxs], -1)
    
#     b_idxs = torch.arange(idxs.shape[0], device=idxs.device)[:, None].repeat(1, idxs.shape[1])

#     return b_idxs.flatten(), idxs.flatten()

def sample_rays_with_distribution(H, W, probs, sample_num, num_view, device):
    
    if probs != None:
        # probs = transforms.GaussianBlur(kernel_size=11, sigma=1.0)(probs.permute(0,3,1,2)).permute(0,2,3,1)
        probs = probs.reshape(probs.shape[0],-1)
        idxs = probs.multinomial(num_samples=sample_num)
        # idxs = []
        # for i in range(num_view):
        #     prob = probs[i].detach().cpu().numpy().flatten()
        #     prob = prob / prob.sum()
        #     idxs += [np.random.choice(H*W, size=(sample_num,), p=prob)]
            
            # idxs += [torch.randperm(H*W, device=device)[:sample_num]]
        # idxs = torch.stack(idxs)
        # idxs = torch.from_numpy(np.stack(idxs)).to(device)
    else:
        idxs = []
        for i in range(num_view):
            idxs += [torch.randperm(H*W, device=device)[:sample_num]]
        idxs = torch.stack(idxs)
        
    b_idxs = torch.arange(idxs.shape[0], device=idxs.device)[:, None].repeat(1, idxs.shape[1])
    
    return b_idxs.flatten(), idxs.flatten()
    


# def gen_patch_ray(H, W, num_view, patch_size, sample_num, device):
    
#     num_patch = int(sample_num // (patch_size ** 2))
#     assert num_patch >= 1
    
#     # w_idx = random.randint(0,W-patch_size)
#     # h_idx = random.randint(0,H-patch_size)
    
#     start_idx = []
#     # probs[:,-(H-patch_size):,-(W-patch_size):, :] = 0
#     for i in range(num_view):
#         start_idx += [np.random.choice((H-patch_size)*(W-patch_size), size=(num_patch,), p=None)]
#     start_idx = torch.from_numpy(np.stack(start_idx))
    
    
#     start_idx = (start_idx // (W - patch_size) * W) + (start_idx % (W - patch_size))
    
#     # num_patch 
#     # w_idx = torch.randint(0,W-patch_size, size=(num_patch,))
#     # h_idx = torch.randint(0,H-patch_size, size=(num_patch,))
#     # # num_patch 
#     # start_idx = h_idx * W + w_idx
    
#     # patch_size x patch_size
#     M = torch.arange(patch_size)[None,:].repeat(patch_size, 1)
#     # num_view x num_patch x patch_size x patch_size
#     M = (M + M.transpose(1,0) * W)[None, None,...] + start_idx[..., None, None]
#     M = M.to(device)
#     b_idxs = torch.arange(num_view, device=device)[:, None].repeat(1, num_patch * patch_size**2)
    
#     return b_idxs.flatten(), M.flatten()


def Rx(theta):
    return np.array([[1, 0, 0], 
                    [0, np.cos(theta), -np.sin(theta)], 
                    [0, np.sin(theta), np.cos(theta)]], 
                    dtype=np.float32)

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], 
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]], 
                    dtype=np.float32)

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], 
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]], 
                    dtype=np.float32)

def adjust_cam(poses):
    poses[:, :, 1] *= -1
    poses[:, :, 2] *= -1

    Rs = poses[:, :3,:3]
    Cs = poses[:, :3, 3]
    angle = (-90, 0, 0)
    mesh_center = np.array([0,0,0])
    scale = 1
    rotation = Rz(angle[2] / 180 * np.pi) @ Ry(angle[1] / 180 * np.pi) @ Rx(angle[0] / 180 * np.pi)
    Cs = scale * ((Cs - mesh_center) @ rotation.transpose())
    Rs = rotation @ Rs 
    poses[:,:3,:3] = Rs 
    poses[:,:3,3] = Cs
    return poses

"""configuration parse 
"""
# class Dict(dict):
#     __setattr__ = dict.__setitem__
#     __getattr__ = dict.__getitem__



# def sample_ray_idx(occlusions, num_patch, ):
#     """
#     occlusions  N x H x W x 1 device tensor 
#     """

#     # N, H, W 
#     torch.where(occlusions.squeeze() == True)


# def get_ray_idx_v2(num_patch, patch_size, occlusion_mask):
#     """
#     num_patch 一张图片需要的patch 数量
#     patch size 
#     occlusion_mask num_camera x H x W x 1
#     """
#     num_camera = occlusion_mask.shape[0]
#     # num_camera x HW
#     occlusion_mask = occlusion_mask.reshape(num_camera, -1)
    
#     cam_idx, pixel_idx = torch.where(occlusion_mask == True)
#     # num_camera 
#     range_pixel = occlusion_mask.float().sum(-1)

#     (torch.rand(num_camera) * range_pixel).int()


def get_ray_idx(idx, patch_size, H, W):
    """
    0 1 2 .. patch_size-1
    1*W 1+1*W ... patch_size-1+1*W
    .
    .
    .
    (patch_size-1)*W .... (patch_size-1)+(patch_size-1)*W
    """
    item1 = torch.arange(patch_size, device=idx.device)[None,:].repeat(patch_size, 1)
    item2 = (torch.arange(patch_size, device=idx.device) * W)[:,None].repeat(1, patch_size)
    offset = item1 + item2  # patch_size x patch_size 

    ray_idx = idx[:, None, None] + offset[None, ...]
    return ray_idx.reshape(-1,)

@torch.no_grad()
def force_binary(x, alpha):
    return 1. / (1 + torch.exp(-alpha * (x - 0.5)))

def binary_loss(T, alpha):
    return nn.MSELoss()(T, force_binary(T, alpha).detach())

def toExtrinsic(c2ws):
    rs = c2ws[:, :3, :3].transpose(0, 2, 1)
    ts = -rs @ c2ws[:, :3, 3:4]
    rts = np.concatenate([rs, ts], axis=-1)
    E = np.zeros((rts.shape[0], 4, 4), dtype=np.float32)
    E[:, -1, -1] = 1
    E[:, :3, :4] = rts 
    return E 

def write_camera(path, idxs, c2ws):
    f = open(path, "w")

    count = 0
    for idx in idxs:
        f.write(f"{idx}\n")
        c2w = c2ws[count]
        f.write(f"{c2w[0,0]} {c2w[0,1]} {c2w[0,2]} {c2w[0,3]}\n{c2w[1,0]} {c2w[1,1]} {c2w[1,2]} {c2w[1,3]}\n{c2w[2,0]} {c2w[2,1]} {c2w[2,2]} {c2w[2,3]}\n")

        count += 1

    f.close()

def load_camera(path):
    f = open(path, "r")

    lines = f.readlines()
    lines = [item.strip("\n") for item in lines]

    func = lambda x: [float(item) for item in x.split(' ')[:4]]

    idxs = []
    c2ws = []
    for i in range(i,len(lines), 4):
        line = lines[i:i+4]
        idx = int(line[0])
        c2w = np.array(list(map(func, line[1:4])))
        idxs += [idx]
        c2ws += [c2w]

    f.close()

    return np.array(c2ws), np.array(idxs)

def stratified_sampling(H, W, num, device, grid_step=100):

    nH = math.ceil(H / grid_step)
    nW = math.ceil(W / grid_step)
    num_per_grid = num // (nH * nW)

    col_idx = torch.randperm(grid_step, device=device)[:num_per_grid]
    row_idx = torch.randperm(grid_step, device=device)[:num_per_grid]

    ray_idx = []
    for i in range(nH):
        for j in range(nW):
            # base_idx = (i * nW + j) * (grid_step**2)
            idxs = (i * grid_step + row_idx) * W + (j * grid_step + col_idx)
            ray_idx += [idxs]
            # indices = torch.randperm(grid_step**2, device=device)[:num_per_grid]
    ray_idx = torch.cat(ray_idx, 0)
    ray_idx = ray_idx[ray_idx < H*W]
    return ray_idx

# if __name__ == "__main__":
#     import cv2 
#     import numpy as np 
#     out = torch.ones((1030, 1030, 3), dtype=torch.float32) * 255
#     ray_idx = stratified_sampling(1030, 1030, 1024, "cpu", 100)
#     out.reshape(-1,3)[ray_idx, 0] = 0 
#     out.reshape(-1,3)[ray_idx, 1] = 0 
#     cv2.imwrite("out.png", out.detach().cpu().numpy())


def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

def parse_yaml(path):
    with open(path, 'r') as f:
        cfg = edict(yaml.full_load(f.read()))
    cfg.yaml = path
    return cfg

def depth_smoothess_loss(depth, color, gamma):
    """
    depth H x W x 1
    color H x W x 3 
    """
    # H-1 x W x 1 
    diff_d_h = torch.abs(depth[1:,...] - depth[:-1,...]) 
    # H-1 x W x 1
    diff_c_h = torch.mean(torch.abs(color[1:,...] - color[:-1, ...]), dim=-1, keepdim=True)

    # H-1 x W x 1
    diff_h = torch.mean(diff_d_h * torch.exp(-gamma * diff_c_h))

    diff_d_w = torch.abs(depth[:,1:,...] - depth[:,:-1,...]) 
    diff_c_w = torch.mean(torch.abs(color[:,1:,...] - color[:,:-1, ...]), dim=-1, keepdim=True)
    diff_w = torch.mean(diff_d_w * torch.exp(-gamma * diff_c_w))
    return diff_h + diff_w 


def split_filename(path):
    for item in path.split('/')[::-1]:
        if item != '':
            return item 
    return None 


    
class Img2Gradient(nn.Module):
    """ compute the gradient of img by pytorch
    """
    def __init__(self, in_channel):
        super(Img2Gradient, self).__init__()
        kh = torch.tensor([[1.,0.,-1.],
                            [2.,0.,-2.],
                            [1.,0.,-1.]], dtype=torch.float32, requires_grad=False)
        kv = torch.tensor([[1.,2., 1.],
                           [0.,0., 0.],
                           [-1.,-2.,-1.]], dtype=torch.float32, requires_grad=False)
        kh = kh.view(1,1,3,3).repeat(1,in_channel,1,1) / 3.
        kv = kv.view(1,1,3,3).repeat(1,in_channel,1,1) / 3.
        self.register_buffer("KH", kh)
        self.register_buffer("KV", kv)
    def forward(self,x):
        xh = F.conv2d(x, self.KH)
        xv = F.conv2d(x, self.KV)
        return (torch.abs(xv) + torch.abs(xh)) / 2.0


def GradLoss(out, gt):
    """
    HWC
    """
    oy = out[1::3, :, :] - out[0::3, :, :]
    ox = out[2::3, :, :] - out[0::3, :, :]
    gy = gt[1::3, :, :] - gt[0::3, :, :]
    gx = gt[2::3, :, :] - gt[0::3, :, :]
    return torch.mean(torch.abs(ox-gx)) + torch.mean(torch.abs(oy-gy))

def Mask_L1Loss(x1, x2, mask):
    item = torch.sum(mask)
    if item != 0:
        return torch.sum(torch.abs(x1-x2) * mask) / item
    else:
        return None 

def Mask_MSELoss(x1, x2, mask):
    item = torch.sum(mask)
    if item != 0:
        return torch.sum(torch.abs(x1-x2)**2 * mask) / item
    else:
        return None

def Mask_huberLoss(x1, x2, mask):
    item = torch.sum(mask)
    
    if item != 0:
        return nn.HuberLoss(reduction='sum', delta=0.5)(x1 * mask, x2 * mask) / item
    else:
        return None 

def compute_hard_mining(x: torch.Tensor, top_rate=0.2):
    N,H,W = x.shape
    flat_x = x.reshape(N, H*W)
    value, idx = flat_x.sort(-1, descending=True) #  N x (HW)
    top_num = int(H*W * top_rate)
    
    idx = idx[:,:top_num]
    t = torch.arange(N).reshape(-1,1).repeat(1,idx.shape[1]).cuda()
    k = torch.cat([t[...,None],idx[...,None]],-1).reshape(-1,2).transpose(1,0)

    masks = torch.zeros(N,H*W)
    masks[k[0],k[1]] = 1.
    masks = masks.reshape(N,H,W)
    masks = masks.cuda()
    return masks, top_num


# def binary_loss(alpha):
#     item = torch.exp(-(alpha-0.5) * 10)
#     return torch.mean(item / ((1. + item)**2))

def hard_mining_loss(pred, target, ratio=0.2):
    """
    pred NCHW
    target NCHW
    """
    mask,top_num = compute_hard_mining(torch.mean(torch.abs(pred - target),1),ratio)
    return torch.sum(torch.mean(torch.abs(pred - target),1) * mask) / top_num
    

def sparsity_loss(sigma, lam):
    return torch.mean(1. - torch.exp(-lam*sigma))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def extract_path(root_dir, epoch=None):
    sorted_func = lambda x:int(os.path.splitext(os.path.basename(x))[0].split('-')[2][1:])
    voxels_path_list = []
    nodes_path_list = []
    models_path_list = []

    exists = []

    try:
        if epoch:
            shared_path = os.path.join(root_dir, f'Coeffi-TShared-E{epoch}.pt')
        else:
            file_list = glob(os.path.join(root_dir, f'Coeffi*.pt'))
            file_list.sort(key=sorted_func)
            shared_path = file_list[-1]
    except:
        shared_path = None 
        print("no shared_path")
    else:
        print(f"shared_path {shared_path}")
    
    file_path = glob(os.path.join(root_dir, "tile-*"))
    
    # print(file_path)

    for path in file_path:

        if epoch:
            nodes_path = glob(os.path.join(path, f'Node*E{epoch}.npy'))
            # nodes_path_list.append(os.path.join(path, f'Node*E{{epoch}}.npy'))
        else:
            nodes_path = glob(os.path.join(path, f'Node*.npy'))

        if len(nodes_path) == 0:
            continue 

        nodes_path.sort(key=sorted_func)

        if epoch:
            voxel_path = glob(os.path.join(path, f'Voxel*E{epoch}.npy'))
        else:
            voxel_path = glob(os.path.join(path, f'Voxel*.npy'))

        voxel_path.sort(key=sorted_func)

        # model_path = glob(os.path.join(path, f'Coeffi*.pt'))
        # try:
        #     model_path.sort(key=sorted_func)
        #     models_path_list.append(model_path[-1])
        # except:
        #     pass

        nodes_path_list.append(nodes_path[-1])
        voxels_path_list.append(voxel_path[-1])
        tileIdx = int(path.split('/')[-1].split('-')[-1])
        exists.append(tileIdx)
    return exists, voxels_path_list, nodes_path_list, models_path_list, shared_path

def extrac_MLP_matrix(model):

    weights = []
    bias = []

    for item in model.mlp:
        if isinstance(item, nn.Linear):
            weights.append(item.weight)
            bias.append(item.bias)
    
    return weights, bias 


def extract_MLP_para(path):
    model_dict = torch.load(path, map_location="cpu")
    weights = []
    bias = []
    for key in model_dict.keys():
        if 'weight' in key:
            weights.append(model_dict[key])
        elif 'bias' in key:
            bias.append(model_dict[key])
    return weights, bias 



@torch.no_grad()
def trilinear_weight(x):
    """ 计算8个neighbor 以及对应的权重
    Args:
        x B x N x 3
        这里的x是相对量，需要先减去 min_corner 
    """
    device = x.device
    # 8 x 3
    base = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                         [1,1,0],[1,0,1],[0,1,1],[1,1,1]],
                         dtype = torch.int32,
                         device = device)
    x0 = x.int() # B x N x 3 
    # B x N x 8 x 3 
    x_nei = x0[...,None,:] + base[None,None,...]
    x_weight = torch.prod(torch.abs(x[...,None,:] - x_nei.float()), dim=-1)
    return x_nei, x_weight[..., None]

@torch.no_grad()
def getTriNeighbor(pts, min_corner, voxel_size):
    """
    Args:
        pts B x N x 3
        min_corner 3
        voxel_size float
    """
    x = (pts - min_corner) / voxel_size
    # B x N x 8 x 3  B x N x 8 x 1
    neighbors_idxs, weights = trilinear_weight(x)
    neighbors = neighbors_idxs * voxel_size + min_corner
    return neighbors, weights

def create_meshgrid(D,H,W):
    Zs,Ys,Xs = torch.meshgrid(torch.arange(D), torch.arange(H),  torch.arange(W))
    return torch.stack([Xs,Ys,Zs], dim=-1)


def get_image(valid, rgb, H, W):

    out = torch.zeros((H*W,3), dtype=torch.float32, device=rgb.device)
    out[valid] = rgb 
    out = out.cpu().numpy().reshape(H,W,3)
    return out 

def get_image_v2(rgb, H, W):
    rgb = rgb.detach().cpu().numpy().reshape(H,W,3)
    return rgb 


def cal_psnr(I1,I2):
    mse = torch.mean((I1-I2)**2)
    if mse < 1e-10:
        return 100
    return 10 * float(torch.log10(255.0**2/mse))



"""Bezier, a module for creating Bezier curves.
Version 1.1, from < BezierCurveFunction-v1.ipynb > on 2019-05-02
"""

class Bezier():
    def TwoPoints(t, P1, P2):
        """
        Returns a point between P1 and P2, parametised by t.
        INPUTS:
            t     float/int; a parameterisation.
            P1    numpy array; a point.
            P2    numpy array; a point.
        OUTPUTS:
            Q1    numpy array; a point.
        """

        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        """
        Returns a list of points interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoints    list of numpy arrays; points.
        """
        newpoints = []
        #print("points =", points, "\n")
        for i1 in range(0, len(points) - 1):
            #print("i1 =", i1)
            #print("points[i1] =", points[i1])

            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
            #print("newpoints  =", newpoints, "\n")
        return newpoints

    def Point(t, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoint     numpy array; a point.
        """
        newpoints = points
        #print("newpoints = ", newpoints)
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
            #print("newpoints in loop = ", newpoints)

        #print("newpoints = ", newpoints)
        #print("newpoints[0] = ", newpoints[0])
        return newpoints[0]

    def Curve(t_values, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t_values     list of floats/ints; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            curve        list of numpy arrays; points.
        """

        if not hasattr(t_values, '__iter__'):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if len(t_values) < 1:
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if not isinstance(t_values[0], (int, float)):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            #print("curve                  \n", curve)
            #print("Bezier.Point(t, points) \n", Bezier.Point(t, points))

            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)

            #print("curve after            \n", curve, "\n--- --- --- --- --- --- ")
        curve = np.delete(curve, 0, 0)
        #print("curve final            \n", curve, "\n--- --- --- --- --- --- ")
        return curve


class PairRandomCrop(object):
    """
    for (lr, hr) pair random crop
    """

    def __init__(self, lr_shape, lr_crop_shape, scale):
        self.lr_h, self.lr_w = lr_shape  
        self.hr_h, self.hr_w = [scale * x for x in lr_shape] 
        self.lr_crop_h, self.lr_crop_w = lr_crop_shape # 64 x 64
        self.hr_crop_h, self.hr_crop_w = self.lr_crop_h * scale, self.lr_crop_w * scale # 128 x 128
        self.lr_h_crop_start = random.randint(
            0, self.lr_h - self.lr_crop_h - 1)
        self.lr_w_crop_start = random.randint(
            0, self.lr_w - self.lr_crop_w - 1)
        self.hr_h_crop_start, self.hr_w_crop_start = self.lr_h_crop_start * \
            scale, self.lr_w_crop_start * scale

    def crop_with_hr_params(self, inputs):
        """
        inputs's shape: (N)HWC
        """
        outputs = inputs[
            ..., 
            self.hr_h_crop_start:self.hr_h_crop_start+self.hr_crop_h,
            self.hr_w_crop_start:self.hr_w_crop_start+self.hr_crop_w,
            :
        ]
        return outputs

    def crop_with_lr_params(self, inputs):
        """
        inputs's shape: (N)hwC
        """
        outputs = inputs[
            ...,
            self.lr_h_crop_start:self.lr_h_crop_start+self.lr_crop_h,
            self.lr_w_crop_start:self.lr_w_crop_start+self.lr_crop_w,
            :
        ]
        return outputs
    
class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)
        
        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        
    def __call__(self, img, scale=1):
        if len(img.shape) == 3:
            return img[self.h1 * scale: self.h2 * scale, self.w1 * scale: self.w2 * scale, :]
        else:
            return img[self.h1 * scale: self.h2 * scale, self.w1 * scale: self.w2 * scale, ...]


def get_grid(data, height, width, normalize=False):
    grid = compute_grid(data, height, width)
    grid = np.frombuffer(grid, dtype=np.float32).reshape((height,width,2))
    if normalize:
        grid = grid / [width-1, height-1] * 2 - 1
    return grid 

def warp_C(K_src, K_dst, E_src, E_dst, D_dst, height, width, normalize=False):
    data = np.concatenate((K_src, E_src[:3,:3].reshape((-1,)), E_src[:3,3:4].reshape((-1,)),
                           K_dst, E_dst[:3,:3].reshape((-1,)), E_dst[:3,3:4].reshape((-1,)),
                           D_dst.reshape((-1,)))).astype(np.float32)

    grid = compute_grid(data.tostring(), height, width)
    grid = np.frombuffer(grid, dtype=np.float32).reshape((height,width,2))
    if normalize:
        grid = grid / [width-1, height-1] * 2 - 1
    return grid

def w2cToc2w(RTs):
    Rs = RTs[:, :3, :3]
    Ts = RTs[:, :3, 3:4]
    Rs = Rs.transpose(0, 2, 1)
    Cs = -np.einsum("ijk, ikl -> ijl", Rs, Ts)
    return np.concatenate([Rs, Cs], -1)
    