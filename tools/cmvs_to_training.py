import numpy as np 
import cv2,os,sys 
import tools, utils 
from glob import glob 
from tqdm import tqdm 
from colmap2cmvs.utils import read_cmvs_cluster

"""
Convert CMVS data to training data 
"""
datadir = sys.argv[1]

outdir = os.path.join(datadir, "clusters")

if os.path.exists(outdir):
    os.system(f"rm -r {outdir}")
os.mkdir(outdir)

intrinsics, RTs, pts, vis = tools.read_bundle(os.path.join(datadir, "bundle.rd.out"))

C2Ws = utils.w2cToc2w(RTs)
focals = intrinsics[:, 0]

clusters, num_cameras, num_clusters = read_cmvs_cluster(os.path.join(datadir, "ske.dat"))
# clusters = [[i for i in range(num_cameras)]]
clusters += [[i for i in range(num_cameras)]]

for idx,cluster in enumerate(clusters):
    cluster.sort()
    if idx == num_clusters:
        cluster_dir = os.path.join(outdir, "all")
    else:
        cluster_dir = os.path.join(outdir, f"{idx}")
    os.mkdir(cluster_dir)
    img_dir = os.path.join(cluster_dir, "images")
    os.mkdir(img_dir)


    depth_list = []
    for cidx in cluster:
        src_path = os.path.join(datadir, "visualize", "%08d"%(cidx) + ".jpg")
        tgt_path = os.path.join(img_dir, "%08d"%(cidx) + ".jpg")
        os.system(f"ln -s {src_path} {tgt_path}")

        points = pts[vis[cidx], :3]
        points = np.concatenate([points, np.ones((points.shape[0],1))], 1)
        z = -np.einsum("ij,kj -> ik", points, RTs[cidx])[:,2]
        depth_list += list(z)
    bds = np.array([np.min(depth_list) * 0.9, np.max(depth_list) * 1.1])
    
    np.save(os.path.join(cluster_dir, "c2ws.npy"), C2Ws[cluster])
    np.save(os.path.join(cluster_dir, "focals.npy"), focals[cluster])
    np.save(os.path.join(cluster_dir, "bds.npy"), bds)



