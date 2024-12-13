import cv2,os,sys 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import os, imageio
import sys 
import math 
sys.path += ["./", "../"]
from tools import tools 
from tools import utils 
import math 
from cfg import *
from scipy.optimize import curve_fit 

# data_dir = "/data/wxc/sig24/data/data/self_data/gardenspheres"
# ks, c2ws = tools.read_campara(os.path.join(data_dir, "renderPath.log"))

# # c2ws[:,1,3] = -1.6
# # tools.write_campara(os.path.join(data_dir, "renderPath.log"), ks, c2ws, 0, 0)
# exit()

LINEAR = False

# data_dir = "/data/wxc/sig24/data/data/self_data/gardenspheres"

# data_dir = "/data/wxc/sig24/data/data/self_data/natatorium"
# data_dir = "/data/wxc/sig24/data/data/self_data/skyscraper"

# data_dir = "/data/wxc/sig24/data/data/self_data/sculpture_dense"
# NUM_FRAME = 600
# data_dir = "/data/wxc/sig24/data/data/self_data/mall"
NUM_FRAME = 240

# data_dir = "/data/wxc/sig24/data/data/self_data/bicycle"

# data_dir = "/data/wxc/sig24/data/data/self_data/red_car"
# data_dir = "/data/wxc/sig24/data/data/self_data/portrait_dense"
# data_dir = "/data/wxc/sig24/data/data/self_data/blue_car_dense"

# data_dir = "/data/wxc/sig24/data/data/self_data/blue_car_blur"
# data_dir = "/data/wxc/sig24/data/data/self_data/bull_new"
data_dir = "/data/wxc/sig24/data/data/self_data/bull_debug5"






def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2))

def complex_path_interpolation(poses, idx_check, num=120):
    num = num // 2
    def gaussian(window_size, sigma):
        gauss = np.array([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def normalize(x):
        return x / np.linalg.norm(x)
    
    poses = poses[idx_check]
    

    
    cs = poses[:,:,3] # N x 3
    
    # N - 1 
    dis = np.linalg.norm(cs[1:,:] - cs[:-1,:], axis=-1)
    dis = np.concatenate([np.zeros((1,)), dis])

    total_dis = np.sum(dis)
    
    step_dis = np.cumsum(dis)
    print(step_dis)
    
    window_size = 10
    # kernel = gaussian(window_size=window_size, sigma=10.0)
    # print(kernel)

    # half_size = window_size//2

    if LINEAR is False:
        s1 = np.linspace(0, 0.5, num//2, endpoint=False)
        s2 = np.linspace(0.5, 0., num//2, endpoint=False)
        s1 = s1 ** 2 * 2
        s2 = s2 ** 2 * 2
        s = np.concatenate([s1, 1. - s2])
    else:
        s = np.linspace(0, 1, num, endpoint=False)
    
    new_poses = np.zeros((num, 3, 4), dtype=np.float32)
    
    index = 0
    for step in s:
        cur_dis = total_dis * step
        ref_idx = next(x[0] for x in enumerate(step_dis) if x[1] > cur_dis)
        
        delta1 = cur_dis - step_dis[ref_idx-1]
        delta2 = step_dis[ref_idx] - cur_dis
        total = delta1 + delta2
        
        delta1 /= total
        delta2 /= total
        
        center = poses[ref_idx-1,:, 3] * delta2 + poses[ref_idx,:, 3] * delta1
        z_axis = poses[ref_idx-1,:, 2] * delta2 + poses[ref_idx,:, 2] * delta1 
        up_axis = poses[ref_idx-1,:, 1] * delta2 + poses[ref_idx,:, 1] * delta1 
        
        # total_weight = 0.0
        # center = np.array([0,0,0], dtype=np.float32)
        # z_axis = np.array([0,0,0], dtype=np.float32)
        # up_axis = np.array([0,0,0], dtype=np.float32)
        # for k in range(window_size):
        #     idx = ref_idx - window_size // 2 + k
            
        #     if ref_idx == idx:
        #         continue
        #     # print(idx, center)
        #     # input()
        #     if idx >= 0 and idx < len(poses)-1:
        #         w = 1
        #         center += poses[idx,:,3] * w 
        #         z_axis += poses[idx,:,2] * w
        #         up_axis += poses[idx,:,1] * w
        #         total_weight += w
        # center = center / total_weight
        # z_axis = z_axis / total_weight
        # up_axis = up_axis / total_weight
        
        # print(center)
        
        x_axis = np.cross(up_axis, z_axis)
        y_axis = np.cross(z_axis, x_axis)
        x_axis = normalize(x_axis)
        y_axis = normalize(y_axis)
        z_axis = normalize(z_axis)
        
        c2w = np.stack([x_axis, y_axis, z_axis, center], axis=-1)
        
        # print(c2w)
        # input()
        new_poses[index] = c2w
        index += 1

    new_poses = np.concatenate([new_poses, new_poses[::-1]], 0)
    points = tools.cameras_scatter(new_poses[:,:3,:3].transpose(0,2,1),
                                  new_poses[:,:3,3],
                                  length=1)
    tools.points2obj(os.path.join(data_dir, "new_render.obj"), points)
    return new_poses

        
    

def compute_interpolate_num(c2w_a, c2w_b):
    ca = c2w_a[:,3]
    cb = c2w_b[:,3]
    distance = np.linalg.norm(ca-cb)
    za = c2w_a[:,2]
    zb = c2w_b[:,2]
    angle = 1 - (za * zb).sum()

    return int(distance / DIS_STEP + angle / ANGLE_STEP)

"""
camera pose interpolation 
"""
"""
give two poses, interpolate
"""
def interpolate_poses(c2w_a, c2w_b, num):
    def normalize(x):
        return x / np.linalg.norm(x)
    poses = np.zeros((num, 3, 4), dtype=np.float32)

    if LINEAR is False:
        s1 = np.linspace(0, 0.5, num//2, endpoint=False)
        s2 = np.linspace(0.5, 0., num//2, endpoint=False)
        s1 = s1 ** 2 * 2
        s2 = s2 ** 2 * 2
        s = np.concatenate([s1, 1. - s2])
    else:
        s = np.linspace(0, 1, num, endpoint=False)

    idx = 0
    for step in s:
        center = c2w_a[:, 3] * (1-step) + c2w_b[:, 3] * step 
        z_axis = c2w_a[:, 2] * (1-step) + c2w_b[:, 2] * step 
        up_axis = c2w_a[:, 1] * (1-step) + c2w_b[:, 1] * step 
        x_axis = np.cross(up_axis, z_axis)
        y_axis = np.cross(z_axis, x_axis)
        x_axis = normalize(x_axis)
        y_axis = normalize(y_axis)
        z_axis = normalize(z_axis)
        c2w = np.stack([x_axis, y_axis, z_axis, center], axis=-1)
        poses[idx] = c2w
        idx += 1
    return poses 


    

def path_interpolation(poses, idx_check, num=120):
    poses = poses[idx_check]
    inter_num = math.ceil(num / (len(poses)-1))
    print(inter_num)
    render_poses = []
    count = 0
    for i in range(0, len(poses)-1):
        
        render_poses += [interpolate_poses(poses[i], poses[i+1], inter_num)]
        count += inter_num
        # if i == len(poses)-1:
        #     inter_num = num - count

    return np.concatenate(render_poses)[:num]


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, load_depth=False,
               load_point = False):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])

    try:
        # f = open(os.path.join(basedir, "clipping_planes.txt"), "r")
        f = open(os.path.join(basedir, "planes.txt"), "r")
    except:
        pass 
    else:
        lines = f.readlines()
        lines = [item.strip().split(" ") for item in lines]
        lines = [[float(item[0]), float(item[1])] for item in lines]
        # lines = np.array(lines)
        lines = lines[0]
        bds[0,:] = lines[0]
        bds[1,:] = lines[1]
        f.close()

        # bds = lines.transpose()

    # poses = poses.transpose(2,0,1)
    # r = poses[0,:3,:3]
    # c = poses[0,:3,3:4]
    # print(c)
    # t = - r.transpose() @ c 
    # print(r)
    # print(t)
    # # print(poses.shape)
    # exit()
    

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    
    img_name = [os.path.splitext(os.path.basename(item))[0] for item in imgfiles]
    # print(idx_check)
    # exit()
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor


    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, apply_gamma=False)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    
    
    if load_depth:
        depdir = os.path.join(basedir, "depths")
        depfiles = [os.path.join(depdir, f) for f in sorted(os.listdir(depdir)) if f.endswith('npy')]
        deps = [np.load(f) for f in depfiles]
        deps = np.stack(deps, 0)
        
    else:
        deps = None
    
    if load_point:
        ptsarr = np.load(os.path.join(basedir, "pts_arr.npy"))
    else:
        ptsarr = None 


    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs, img_name, deps, ptsarr

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = rads * 0.8
    rads = np.array(list(rads) + [1.])

    # rads[1] *= 0 
    # print(rads)
    # exit()
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        c[1] = c2w[1,3]
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    
    render_poses = np.array(render_poses)
    render_poses = render_poses[:render_poses.shape[0]//2]
    render_poses = np.concatenate([render_poses, render_poses[::-1]], 0)
    return render_poses
    


def recenter_poses(poses, ptsarr, load_point):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    # print(c2w.shape, poses.shape)
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    
    if load_point:
        
        ptsarr = np.concatenate([ptsarr, np.ones_like(ptsarr[...,:1])], -1)
        
        ptsarr = np.sum(np.linalg.inv(c2w)[None, :, :] * ptsarr[:,None, :], axis=-1)
        ptsarr = ptsarr[...,:3]
    
    return poses, ptsarr


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, NUM_FRAME):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds


def get_render_poses(poses, bds):
    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3,:4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = NUM_FRAME
    N_rots = 2
    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)   
    
    return render_poses     

def load_self_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, 
                   rendering_mode=FORWARD_FACING, load_depth=False, load_point=False):
    

    poses, bds, imgs, img_name, deps, ptsarr = \
        _load_data(basedir, factor=factor, load_depth=load_depth, load_point=load_point) # factor=8 downsamples original imgs by 8x
    # bds[0,:] = 1.2 
    print('Loaded', basedir, bds.min(), bds.max())

    
    # Correct rotation matrix ordering and move variable dim to axis 0
    

    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)

    # print(poses[:,:3,3].min(axis=0))
    # print(poses[:,:3,3].max(axis=0))
    # exit()
    # r = poses[0,:3,:3]
    # c = poses[0,:3,3:4]
    # t = -r.transpose() @ c 
    # print(r)
    # print(t)
    # exit()
    
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # hwf = poses[0,:3,-1] * 1.13
    hwf = poses[0,:3,-1] * 1.05
    focal = hwf[2] 
    
    if rendering_mode == FORWARD_FACING:
        file = os.path.join(basedir, "forward.txt")
        if os.path.exists(file):
            with open(os.path.join(basedir, "forward.txt"), "r") as f:
                lines = f.readlines()
                lines = [item.strip() for item in lines]
                idx_check = [img_name.index(item) for item in lines]
        else:
            idx_check = [ _ for _ in range(len(poses))]
        ori_render_poses = get_render_poses(poses[idx_check], bds)
        ori_render_poses = np.array(ori_render_poses).astype(np.float32)
    elif rendering_mode == POSE_INTER:
        with open(os.path.join(basedir, "render.txt"), "r") as f:
            lines = f.readlines()
            lines = [item.strip() for item in lines]
            idx_check = [img_name.index(item) for item in lines]
        ori_render_poses = path_interpolation(poses.copy(), idx_check, NUM_FRAME)
    elif rendering_mode == FOCUS:
        _, ori_render_poses, _ = spherify_poses(poses.copy(), bds)
    else:
        with open(os.path.join(basedir, "render.txt"), "r") as f:
            lines = f.readlines()
            lines = [item.strip() for item in lines]
            idx_check = [img_name.index(item) for item in lines]
        ori_render_poses = complex_path_interpolation(poses.copy(), idx_check, NUM_FRAME)
    
    tools.write_campara_v2(os.path.join(basedir, "render_ori.log"), focal,
                        ori_render_poses[:,:3,:4], images.shape[1], images.shape[2])
    

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if load_depth:
        deps *= sc
    if load_point:
        ptsarr *= sc
    # print(sc)
    # print(poses[0])
    # exit()


    # r = poses[0,:3,:3]
    # c = poses[0,:3,3:4]
    # t = -r.transpose() @ c 
    # t /= sc
    # print(r)
    # print(t)
    # exit()


    ### for 3DGS 
    if rendering_mode == FORWARD_FACING:
        # with open(os.path.join(basedir, "forward.txt"), "r") as f:
        #     lines = f.readlines()
        #     lines = [item.strip() for item in lines]
        #     idx_check = [img_name.index(item) for item in lines]
        GS_render_poses = get_render_poses(poses[idx_check], bds)
        GS_render_poses = np.array(GS_render_poses)
    elif rendering_mode == POSE_INTER:
        GS_render_poses = path_interpolation(poses.copy(), idx_check, NUM_FRAME)
    elif rendering_mode == FOCUS:
        _, GS_render_poses, _ = spherify_poses(poses.copy(), bds)
    else:
        GS_render_poses = complex_path_interpolation(poses.copy(), idx_check, NUM_FRAME)
    



    GS_render_poses[:,:3,1] *= -1
    GS_render_poses[:,:3,2] *= -1
    GS_render_poses[:,:3,3] /= sc


    tools.write_campara_v2(os.path.join(basedir, "render.log"), focal,
                        GS_render_poses[:,:3,:4], images.shape[1], images.shape[2])


    if recenter:
        poses, ptsarr = recenter_poses(poses, ptsarr, load_point)
        
    if rendering_mode == FORWARD_FACING:
        # with open(os.path.join(basedir, "forward.txt"), "r") as f:
        #     lines = f.readlines()
        #     lines = [item.strip() for item in lines]
        #     idx_check = [img_name.index(item) for item in lines]
        render_poses = get_render_poses(poses[idx_check], bds)
        render_poses = np.array(render_poses).astype(np.float32)
    elif rendering_mode == POSE_INTER:
        render_poses = path_interpolation(poses.copy(), idx_check, NUM_FRAME)
    elif rendering_mode == FOCUS:
        _, render_poses, _ = spherify_poses(poses.copy(), bds)
    else:
        render_poses = complex_path_interpolation(poses.copy(), idx_check, NUM_FRAME)
    render_poses[:,:3,1] *= -1
    render_poses[:,:3,2] *= -1
    
    # render_poses[:,:,3] += render_poses[:,:,2] * 1
    # render_poses[:,:,3] -= render_poses[:,:,2] * 5
    # render_poses[:,2,3] += 5
    # render_poses[:,1,3] -=0.5
    # render_poses[:,:,3]

    tools.write_campara_v2(os.path.join(basedir, "render_ours.log"), focal,
                        render_poses[:,:3,:4], images.shape[1], images.shape[2])

    points = tools.cameras_scatter(render_poses[:,:3,:3].transpose(0,2,1),
                                render_poses[:,:3,3],
                                length=1)
    tools.points2obj(os.path.join(data_dir, "cam_render.obj"), points)


# load_self_data(data_dir, factor=4, rendering_mode=POSE_INTER)
# load_self_data(data_dir, factor=4, rendering_mode=None)
load_self_data(data_dir, factor=4, rendering_mode=FORWARD_FACING)


