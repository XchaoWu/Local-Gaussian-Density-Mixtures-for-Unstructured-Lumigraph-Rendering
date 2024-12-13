import numpy as np
import os, sys, cv2
sys.path += ['./', '../']
from poses import colmap_read_model as read_model



def write2bundle(camdata, imdata, pts3d, bundle_path):

    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]

    h, w, f = cam.height, cam.width, cam.params[0]

    if len(cam.params) == 4:
        k1 = cam.params[3]
        k2 = 0.0
    elif len(cam.params) == 5:
        k1 = cam.params[3]
        k2 = cam.params[4]
    else:
        raise NotImplementedError

    bundle_file = open(bundle_path, "w")

    num_cameras = len(imdata)
    num_points = len(pts3d)

    bundle_file.write("# Bundle file v0.3\n")
    bundle_file.write(f"{num_cameras} {num_points}\n")


    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        bundle_file.write(f"{f} {k1} {k2}\n")
        bundle_file.write(f"{R[0,0]} {R[0,1]} {R[0,2]}\n")
        bundle_file.write(f"{R[1,0]} {R[1,1]} {R[1,2]}\n")
        bundle_file.write(f"{R[2,0]} {R[2,1]} {R[2,2]}\n")
        bundle_file.write(f"{t[0,0]} {t[1,0]} {t[2,0]}\n")

    for p in pts3d:
        pt = pts3d[p]
        bundle_file.write(f"{pt.xyz[0]} {pt.xyz[1]} {pt.xyz[2]}\n")
        bundle_file.write(f"{pt.rgb[0]} {pt.rgb[1]} {pt.rgb[2]}\n")
        num_views = len(pt.image_ids)
        bundle_file.write(f"{num_views}")
        for i in range(num_views):
            img_id = pt.image_ids[i]
            p2d_id = pt.point2D_idxs[i]
            xy = imdata[img_id].xys[p2d_id]
            bundle_file.write(f" {img_id} {p2d_id} {xy[0]} {xy[1]}")
        bundle_file.write("\n")

    bundle_file.close()


def load_colmap_data(data_dir):

    camerasfile = os.path.join(data_dir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    imagesfile = os.path.join(data_dir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    points3dfile = os.path.join(data_dir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    return camdata, imdata, pts3d



