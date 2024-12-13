import cv2,os 
import torch 
import numpy as np
from glob import glob 
from tqdm import tqdm 
from dataloader.load_llff_new import load_llff_data
from dataloader.load_self_data import load_self_data
from pinhole_camera import PinholeCamera
from easydict import EasyDict as edict
from cfg import * 


def load_data(data_dir, train_height, train_width, data_type, factor=1, load_point=False, 
              rendering_mode=FORWARD_FACING):
    assert data_type in [LLFF, SELFDATA]

    out = {}
    
    if data_type == LLFF:
        images, poses, bds, render_poses, i_test, intrinsic, dmin, dmax = \
                load_llff_data(data_dir, height=train_height, width=train_width)
        poses[:,:3,1] *= -1
        poses[:,:3,2] *= -1
        render_poses[:,:3,1] *= -1
        render_poses[:,:3,2] *= -1
        # hwf = poses[0,:3,-1]
        intrinsic = intrinsic.flatten()
        H = images.shape[1]; W = images.shape[2]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, intrinsic[2:])
        # if not isinstance(i_test, list):
        #     i_test = [i_test]
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test)])

        print('DEFINING BOUNDS')
        # if args.no_ndc:
        # near = np.ndarray.min(bds) * .9
        # far = np.ndarray.max(bds) * 1.
        near = dmin 
        far = dmax
        
        print('NEAR FAR', near, far)

        ks = torch.tensor([intrinsic[2], 0, intrinsic[4], 
                        0, intrinsic[3], intrinsic[5], 
                        0., 0., 1.], 
                        dtype=torch.float32).reshape(1,3,3).repeat(len(images),1,1).numpy()
        c2ws = poses
        render_ks = torch.tensor([intrinsic[2], 0, intrinsic[4], 
                        0, intrinsic[3], intrinsic[5], 
                        0., 0., 1.], 
                        dtype=torch.float32).reshape(1,3,3).repeat(render_poses.shape[0],1,1).numpy()

        out.update({"near": near, "far": far})


    elif data_type == SELFDATA:
        images, poses, bds, render_ks, render_poses, i_test, deps, ptsarr = load_self_data(data_dir, factor=factor, 
                                                                  rendering_mode=rendering_mode,
                                                                  load_point=load_point)
        poses[:,:3,1] *= -1
        poses[:,:3,2] *= -1
        render_poses[:,:3,1] *= -1
        render_poses[:,:3,2] *= -1
        hwf = poses[0,:3,-1]
        focal = hwf[2]
        H = images.shape[1]; W = images.shape[2]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, focal)
        # if not isinstance(i_test, list):
        #     i_test = [i_test]
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test)])

        print('DEFINING BOUNDS')
        # if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
        # near = dmin 
        # far = dmax
        
        print('NEAR FAR', near, far)

        ks = torch.tensor([focal, 0, W/2., 
                        0, focal, H/2., 
                        0., 0., 1.], 
                        dtype=torch.float32).reshape(1,3,3).repeat(len(images),1,1).numpy()
        c2ws = poses

        out.update({"near": near, "far": far, "deps": deps, "ptsarr":ptsarr})
    

    out.update({"ks": ks, "c2ws": c2ws, "images": images,
                "i_train": i_train, "i_test": i_test,
                "render_ks":render_ks, "render_c2ws": render_poses, "H": H, "W": W})
    return edict(out) 
    
if __name__ == "__main__":
    pass 