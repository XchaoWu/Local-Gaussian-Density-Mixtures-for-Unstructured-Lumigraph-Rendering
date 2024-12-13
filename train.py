import os,sys,cv2 
import numpy as np 
import torch 
import torch.nn as nn  
import random 
from LGDM import LGDM 
from scheduler import Scheduler, SchedulerManager
from pinhole_camera import PinholeCamera
from tqdm import tqdm 
from tools import tools,utils
from cfg import * 
from datetime import datetime
from HashGrid import HashGrid
from ray_sampler import RaySample
import network 
import time 
from glob import glob 
from load_data import load_data 
from occupied_grid import OccupiedGrid
from easydict import EasyDict as edict
from criterion import Criterion
from ssim import SSIM
import torch.nn.functional as F
# from skimage.metrics import peak_signal_noise_ratio, structure_similarity
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from cuda import padding_results_cuda 

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

gpuIdx = int(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpuIdx}"
device = torch.device("cuda:0")

"""
python train.py [GPU_IDX] [config.yaml] [data_name] [log_name]

"""


cfg = utils.parse_yaml(sys.argv[2])
data_name = sys.argv[3]
PS = sys.argv[4]

num_neighbor = cfg.NUM_NEIGHBOR
num_gaussian = cfg.NUM_GAUSSIAN
rendering_mode = FORWARD_FACING 
factor = cfg.FACTOR

DATA_TYPE = cfg.DATA_TYPE
SSIM_SCIKIT = cfg.SSIM_SCIKIT

try:
    WEIGHT_INTER = cfg.WEIGHT_INTER
except:
    WEIGHT_INTER = 1.0

try:
    WEIGHT_PER_VIEW = cfg.WEIGHT_PER_VIEW
except:
    WEIGHT_PER_VIEW = 1.0

try:
    WEIGHT_VOTING = cfg.WEIGHT_VOTING
except:
    WEIGHT_VOTING = 0.01
    
try:
    INV_Z = cfg.INV_Z
except:
    INV_Z = False
print("INV_Z", INV_Z)

print(f"WEIGHT_INTER {WEIGHT_INTER} WEIGHT_PER_VIEW {WEIGHT_PER_VIEW} WEIGHT_VOTING {WEIGHT_VOTING}\n")

if DATA_TYPE == LLFF:
    data_name += "_undistort"
data_dir = os.path.join(cfg.DATA_DIR, data_name)


batch_size = cfg.BATCH_SIZE
total_step = cfg.TOTAL_STEP

num_sample = cfg.NUM_SAMPLE
PREFIX = "TEST"

DEBUG = False

init_bound = 1.0
padding_results = True

start_eta = cfg.ETA[0]
end_eta = cfg.ETA[1]

hierarchical_sampling = True 

inference_image_step = 2000
inference_per_view = 20000
max_grid_log2dim = cfg.MAX_GRID_LOG2DIM
voting_decay = True
inference_video_step = [10000, total_step]
export_step = [10000, 20000, 30000, total_step]

if DATA_TYPE == LLFF:
    data = load_data(data_dir, cfg.TRAIN_HEIGHT, cfg.TRAIN_WIDTH, DATA_TYPE, factor=factor, 
                 rendering_mode=rendering_mode)
else:
    data = load_data(data_dir, None, None, DATA_TYPE, factor=factor, 
                 rendering_mode=rendering_mode)


H = data.H; W = data.W

i_train = data.i_train
i_test = data.i_test
images = data.images

train_images = images[i_train]
test_images = images[i_test]
try:
    SPARSITY = cfg.SPARSITY
except:
    pass 
else:
    i_train = i_train[0::SPARSITY]
    
    print(f"======= SPARSITY {SPARSITY} ==============")

camera = PinholeCamera(H, W, 
                    ks=torch.from_numpy(data.ks[i_train]).to(device), 
                    c2ws=torch.from_numpy(data.c2ws[i_train]).to(device))

test_camera = PinholeCamera(H, W, 
                            ks=torch.from_numpy(data.ks[i_test]).to(device), 
                            c2ws=torch.from_numpy(data.c2ws[i_test]).to(device))

render_camera = PinholeCamera(H, W, 
                            ks=torch.from_numpy(data.render_ks).to(device), 
                            c2ws=torch.from_numpy(data.render_c2ws[:,:3,:4]).to(device))



points = tools.cameras_scatter(data.render_c2ws[:,:3,:3].transpose(0,2,1),
                              data.render_c2ws[:,:3,3],
                              length=1)
tools.points2obj(os.path.join(data_dir, "cam_render.obj"), points)
points = tools.cameras_scatter(data.c2ws[:,:3,:3].transpose(0,2,1),
                              data.c2ws[:,:3,3],
                              length=1)

tools.points2obj(os.path.join(data_dir, "cam_train.obj"), points)

if DATA_TYPE == LLFF:    
    ogrid = OccupiedGrid((3,3,3), device)
    near = data.near 
    far = data.far
    ogrid.near = near 
    ogrid.far = far
    ogrid.init_with_nearfar(camera, near, far)
elif DATA_TYPE == SELFDATA:
    # forward facing data 
    ogrid = OccupiedGrid((3,3,3), device)
    near = data.near 
    far = data.far
    ogrid.near = near 
    ogrid.far = far
    ogrid.init_with_nearfar(camera, near, far)
    # ogrid.init_grid_cover_center(camera, near, far)
else:
    raise NotImplementedError 

ogrid.inv = INV_Z

if DEBUG == False:
    runtime = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if PREFIX != "":
        log_dir = os.path.join(data_dir, "logs", f"{PREFIX}-{PS}-{runtime}")
    else:
        log_dir = os.path.join(data_dir, "logs", f"{PS}-{runtime}")
else:
    log_dir = os.path.join(data_dir, "logs", f"DEBUG")

if os.path.exists(log_dir) is False:
    # os.system(f"rm -rf {log_dir}")
    os.makedirs(log_dir)

os.system(f"cp {cfg.yaml} {log_dir}")
    
save_dir = os.path.join(log_dir, f"per-view")
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
check_dir = os.path.join(log_dir, "checkpoint")
if os.path.exists(check_dir) is False:
    os.mkdir(check_dir)
video_dir = os.path.join(log_dir, "video")
if os.path.exists(video_dir) is False:
    os.mkdir(video_dir)
final_pred_dir = os.path.join(log_dir, "final_pred")
if os.path.exists(final_pred_dir) is False:
    os.mkdir(final_pred_dir)
gt_dir = os.path.join(final_pred_dir, "gt")
if os.path.exists(gt_dir) is False:
    os.mkdir(gt_dir)
pred_dir = os.path.join(final_pred_dir, "renders")
if os.path.exists(pred_dir) is False:
    os.mkdir(pred_dir)
    

model = LGDM(train_images.shape[0], train_images,
             num_neighbor, num_gaussian, DATA_TYPE, hierarchical_sampling, device)
model.set_ogrid(ogrid)


train_images = torch.from_numpy(train_images).float() 

for i in range(len(test_images)):
    cv2.imwrite(os.path.join(gt_dir, f"{i}.png"), test_images[i][...,::-1] * 255)
    
params = []


params += [{"params": model.parameters(), "lr": start_eta}]

optimizer = torch.optim.Adam(params) 


scheduler_list = [Scheduler("all", start_eta=start_eta, end_eta=end_eta, iterations=total_step, groups=[])]
Sche = SchedulerManager(scheduler_list)


cri = Criterion()
cri.append(nn.MSELoss(), "INTER_RGB_LOSS", weight=WEIGHT_INTER)
cri.append(nn.MSELoss(), "PER_VIEW_RGB_LOSS", weight=WEIGHT_PER_VIEW)



def voting_loss(weight,ref_weight,valid):

    # KL divergence B x 1 s
    kl_loss = torch.sum(weight * ( weight / (ref_weight + 1e-8) + 1e-8).log(), dim=1)
    
    final_loss = kl_loss
    
    final_loss = torch.sum(final_loss * valid) / (torch.sum(valid) + 1e-8)
    
     
    return final_loss


cri.append(voting_loss, "VOTING_LOSS", weight=WEIGHT_VOTING)


def decay_weight(global_step):
    return 0.01 * (0.1 ** (global_step / total_step))


RS = RaySample(camera.num_camera, H, W, device)

patch_size = 2
for global_step in tqdm(range(1, total_step+1)):
    
    

    if voting_decay and WEIGHT_VOTING > 0:
        cri.loss_items["VOTING_LOSS"].weight = decay_weight(global_step)
    

    b_idx, ray_idx = RS.sample_rays(batch_size, num_neighbor)

    rays_o, rays_d, up_axis = camera.get_rays(ray_idx=(b_idx, ray_idx))

    
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    up_axis = up_axis.reshape(-1,3)


    gt_color= train_images.reshape(-1, H*W, 3)[b_idx, ray_idx]
    gt_color = gt_color.reshape(-1,3).to(device)



    out = model.render_rays(rays_o, rays_d, camera, num_sample, TRAIN,
                            global_step, total_step, ref_idxs=b_idx.int(), ray_idx=ray_idx.int(),
                            up_axis=up_axis)

        
    pred_color = out["pred_color"]
    
    pred_depth = out["pred_depth"]

    pred_color_ref = out["pred_color_ref"]

    pred_depth_ref = out["pred_depth_ref"]
    
    valid = out["valid"]


    nei_idxs = out["nei_idxs"]
    warped_uvs  = out["warped_uvs"]

    RS.update(b_idx, ray_idx, pred_color_ref*valid, pred_color*valid, gt_color*valid)
    
    
    model.valid_mask.reshape(-1, H*W, 1)[b_idx, ray_idx, :] = valid

    loss = 0

    if WEIGHT_INTER > 0:
        interpolation_loss = cri.compute_loss("INTER_RGB_LOSS", pred_color * valid, gt_color * valid, global_step)
        loss += interpolation_loss
    
    if WEIGHT_VOTING > 0:
        loss += cri.compute_loss("VOTING_LOSS", out["weight"], out["ref_weight"], 
                                                global_step, valid=valid)
    
    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    



    if global_step == 1 or global_step % 100 == 0:
        
        info = f"\n======GPU {gpuIdx} Data {data_name} Note {PS}======\n"
        info += Sche.getInfo()
        info += cri.getInfo()
        
        # info += f"gausssian prunning weight {model.gausssian_prunning_weight}\n"

        print(info)


    if global_step % inference_image_step == 0:
        mean_psnr = 0.0
        mean_ssim = 0.0

        for k in tqdm(range(0,test_camera.num_camera)):
            rays_o, rays_d, up_axis = test_camera.get_rays(view_idx=k)
            rays_o = rays_o.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)
            up_axis = up_axis.reshape(-1,3)
            rgb = torch.zeros_like(rays_o)
            valid = torch.zeros_like(rays_o[...,:1]).bool()
            # grgb = torch.zeros_like(rays_o)
            depth = torch.zeros_like(rays_o[...,:1])
            bs = 2**11

            with torch.no_grad():
                for i in range(0, rgb.shape[0], bs):
                    out = model.render_rays(rays_o[i:i+bs], rays_d[i:i+bs], camera, num_sample, INFERENCE,
                                                global_step, total_step, up_axis=up_axis)
                    rgb[i:i+bs] = out["pred_color"]
                    # grgb[i:i+bs] = out["pred_global_color"]
                    # diffuse[i:i+bs] = out["pred_diffuse"]
                    depth[i:i+bs] = out["pred_depth"]
                    valid[i:i+bs] = out["valid"]
                    
            if padding_results:
                padding_results_cuda(valid.reshape(test_camera.H,test_camera.W,1), 
                                     rgb.reshape(test_camera.H,test_camera.W,3))
            
            rgb = rgb.detach().cpu().numpy().reshape(test_camera.H,test_camera.W,3) 
            # grgb = grgb.detach().cpu().numpy().reshape(H,W,3)
            depth = depth.detach().cpu().reshape(test_camera.H,test_camera.W,1).repeat(1,1,3).numpy()
            
            depth = ((depth - data.near) / (data.far - data.near)).clip(0,1)

            gt_images = test_images[k]


            test_psnr = peak_signal_noise_ratio(rgb, gt_images, data_range=1.0)
            
            if SSIM_SCIKIT:
                test_ssim = structural_similarity(rgb, gt_images, win_size=11, 
                                                multichannel=True, gaussian_weights=True)
            else:
                test_ssim = float(utils.ssim(torch.from_numpy(rgb)[None,...].permute(0,3,1,2).double(),
                                            torch.from_numpy(gt_images)[None,...].permute(0,3,1,2).double()))

            # test_psnr = utils.cal_psnr(torch.from_numpy(rgb), torch.from_numpy(gt_images))
            
            # test_ssim = 1. - SSIM_CAL(torch.from_numpy(rgb).permute(2,0,1)[None,...] / 255., 
            #                         torch.from_numpy(gt_images).permute(2,0,1)[None,...] / 255.)
            print(f"test PSNR: {test_psnr:.3f}\ttest SSIM: {test_ssim:.3f}")
            # out1 = np.concatenate([gt_images, depth], axis=1)
            # out2 = np.concatenate([grgb, grgb], axis=1)
            # out = np.concatenate([out1, out2], axis=0) * 255
            out = np.concatenate([gt_images,rgb,depth], axis=1) * 255
            cv2.imwrite(os.path.join(log_dir,f"pred-{k}-{global_step}-{test_psnr:.3f}-{test_ssim:.3f}.png"),  out[..., ::-1])
            
            cv2.imwrite(os.path.join(pred_dir, f"{k}.png"), rgb[...,::-1]*255)
            
            mean_psnr += test_psnr
            mean_ssim += test_ssim
        mean_psnr = mean_psnr / test_camera.num_camera
        mean_ssim = mean_ssim / test_camera.num_camera
        
        print(f"mean PSNR: {mean_psnr:.3f}\tmean SSIM: {mean_ssim:.3f}")
        
        with open(os.path.join(log_dir, "score.log"), "a") as f:
            f.write(f"Global step {global_step}\nmean PSNR: {mean_psnr:.2f}\tmean SSIM: {mean_ssim:.3f}\n\n")
        
        # model.ogrid.vis_grid(os.path.join(log_dir,f"grid.obj"))
        
    if  global_step % inference_per_view == 0:

        model.update_rendered_per_view_depth(camera, num_sample)
        # model.ogrid.vis_grid(os.path.join(log_dir,f"grid.obj"))
        
        torch.cuda.empty_cache()

        for idx in tqdm(range(camera.num_camera)):
            dep = model.per_view_depth[idx].detach().cpu().reshape(H,W,1).repeat(1,1,3).numpy()
            dep = ((dep - data.near) / (data.far - data.near)).clip(0,1) * 255 
            img = model.images[idx].detach().cpu().numpy()
            
            pimg_ref = RS.progress_img_ref[idx]
            pimg = RS.progress_img[idx]
            err = RS.error_map[idx].cpu().repeat(1,1,3).numpy() * 255
            err_ref = RS.error_map_ref[idx].cpu().repeat(1,1,3).numpy() * 255
            
            vmask = model.valid_mask[idx].float().repeat(1,1,3).cpu().numpy() * 255
            row1 = np.concatenate([img, pimg_ref, pimg], 1)
            row2 = np.concatenate([dep, err_ref, err,], 1)
            out = np.concatenate([row1, row2], 0)
            cv2.imwrite(os.path.join(save_dir,f"view-{idx}.png"), out[..., ::-1])
        
    if hierarchical_sampling and model.ogrid.log2dim_resolution.min() < max_grid_log2dim and global_step % 1000 == 0:
        model.ogrid.split(max_log2dim=max_grid_log2dim)
        model.ogrid.remove_invisiable(camera)
    
    # if global_step % 1000 == 0:
    #     model.ogrid.update_mask()
        
    if global_step in export_step:
        model.export(check_dir)
        
        # export cam 
        np.savez(os.path.join(check_dir, "cam.npz"), 
                 c2ws=camera.get_poses().detach().cpu().numpy(),
                 ks=camera.ks.detach().cpu().numpy(),
                 test_c2ws=test_camera.get_poses().detach().cpu().numpy(),
                 test_ks=test_camera.ks.detach().cpu().numpy(),
                 render_c2ws=render_camera.get_poses().detach().cpu().numpy(),
                 render_ks=render_camera.ks.detach().cpu().numpy(),
                 near=near, far=far)
        
    if global_step in inference_video_step:
        img_list = []
        torch.cuda.empty_cache()
        for k in tqdm(range(0,render_camera.num_camera,1)):
            rays_o, rays_d, up_axis = render_camera.get_rays(view_idx=k)
            rays_o = rays_o.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)
            up_axis = up_axis.reshape(-1,3)
            rgb = torch.zeros_like(rays_o)
            valid = torch.zeros_like(rays_o[...,:1]).bool()
            dep = torch.zeros_like(rays_o[...,:1])
            bs = 2**11
            with torch.no_grad():
                for i in range(0, rgb.shape[0], bs):
                    out = model.render_rays(rays_o[i:i+bs], rays_d[i:i+bs], camera, num_sample, INFERENCE,
                                                global_step, total_step, up_axis=up_axis)
                    rgb[i:i+bs] = out["pred_color"]
                    dep[i:i+bs] = out["pred_depth"]
                    valid[i:i+bs] = out["valid"]
            
            if padding_results:
                padding_results_cuda(valid.reshape(render_camera.H,render_camera.W,1), rgb.reshape(render_camera.H,render_camera.W,3))
            rgb = rgb.detach().cpu().numpy().reshape(render_camera.H,render_camera.W,3) * 255
            dep = dep.detach().cpu().reshape(render_camera.H,render_camera.W,1).repeat(1,1,3).numpy()
            dep = ((dep - data.near) / (data.far - data.near)).clip(0,1) * 255 
            res = np.concatenate([rgb, dep], axis=1)
            
            # print(dep.min(), dep.max())
            
        #     vwriter.write(res[..., ::-1].astype(np.uint8))
        # vwriter.release()
            cv2.imwrite(os.path.join(video_dir, f"{k}.png"), rgb[..., ::-1])
            # exit()
            # break
            img_list += [res]
            
        tools.generate_video(os.path.join(log_dir,f"video-{global_step}.mp4"), img_list, fps=30)
        
        # tools.save_img_list(log_dir, img_list)
        
        input_path = os.path.join(video_dir, "%d.png")
        output_path = os.path.join(log_dir,f"video-ffmpeg-{global_step}.mp4")
        os.system(f"ffmpeg -f image2 -framerate 30 -i {input_path} -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -vcodec libx264 -pix_fmt yuv420p {output_path}")


    Sche.step(global_step, optimizer)