import numpy as np
import os, sys, cv2
from tqdm import tqdm 
sys.path += ['./', '../']
from colmap2bundle import write2bundle, load_colmap_data
import utils 
import tools 

data_dir = '/data/wxc/sig24/data/sig23/playground/temp'


image_dir = os.path.join(data_dir, "images")
out_dir = os.path.join(data_dir, "cmvs_format")

if os.path.exists(out_dir) is False:
    os.mkdir(out_dir)

bundle_path = os.path.join(out_dir, "bundle.rd.out")
vis_dir = os.path.join(out_dir, "visualize")
txt_dir = os.path.join(out_dir, "txt")
perview_dir = os.path.join(out_dir, "perview")
poses_dir = os.path.join(out_dir, "poses")

if os.path.exists(vis_dir) is False:
    os.mkdir(vis_dir)
if os.path.exists(txt_dir) is False:
    os.mkdir(txt_dir)
if os.path.exists(perview_dir) is False:
    os.mkdir(perview_dir)
if os.path.exists(poses_dir) is False:
    os.mkdir(poses_dir)

camdata, imdata, pts3d = load_colmap_data(data_dir)

list_of_keys = list(camdata.keys())
cam = camdata[list_of_keys[0]]

h, w, f = cam.height, cam.width, cam.params[0]

K = np.array([f, 0, cam.params[1], 0, f, cam.params[2], 0, 0, 1], dtype=np.float32).reshape(3,3)

write2bundle(camdata, imdata, pts3d, bundle_path)

for idx,k in enumerate(imdata):
    im = imdata[k]
    name = im.name 
    print(name, idx)


    f = open(os.path.join(perview_dir, "%08d"%(idx) + ".txt"), "w")
    f.write(' '.join([str(item) for item in im.point3D_ids if item != -1]) + "\n")
    f.close()   
    # im.point3D_ids

    img = cv2.imread(os.path.join(image_dir, name))
    cv2.imwrite(os.path.join(vis_dir, "%08d"%(idx) + ".jpg"), img)

    R = im.qvec2rotmat()
    t = im.tvec.reshape([3,1])

    RT = np.concatenate([R,t], 1)


    f = open(os.path.join(poses_dir, "%08d"%(idx) + ".txt"), "w")
    f.write("CONTOUR\n")
    f.write(f"{RT[0,0]} {RT[0,1]} {RT[0,2]} {RT[0,3]}\n")
    f.write(f"{RT[1,0]} {RT[1,1]} {RT[1,2]} {RT[1,3]}\n")
    f.write(f"{RT[2,0]} {RT[2,1]} {RT[2,2]} {RT[2,3]}\n")
    f.close()

    CAM = K @ RT 

    f = open(os.path.join(txt_dir, "%08d"%(idx) + ".txt"), "w")
    f.write("CONTOUR\n")
    f.write(f"{CAM[0,0]} {CAM[0,1]} {CAM[0,2]} {CAM[0,3]}\n")
    f.write(f"{CAM[1,0]} {CAM[1,1]} {CAM[1,2]} {CAM[1,3]}\n")
    f.write(f"{CAM[2,0]} {CAM[2,1]} {CAM[2,2]} {CAM[2,3]}\n")
    f.close()
