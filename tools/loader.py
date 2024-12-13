import numpy as np 
import sys,os 
from glob import glob 



# =================================== Tanks and temples Data ===================================

def load_camera_pose_tat(file):
    """
    load camera from data tanks and temples
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    c2w = np.array(list(map(lambda x:float(x),lines[0].strip().split(' ')))[:-4]).reshape(3,4)
    return c2w 

def load_intrinsic_tat(file):
    """
    load camera intrinsic from data tanks and temples
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    K = np.array(list(map(lambda x: float(x), lines[0].strip().split(' ')))).reshape(4,4)
    return K[:3,:3]

def load_cameras_tat(path):
    """
    batch load camera paras from data tanks and temples
    """
    pose_dir = os.path.join(path, "pose")
    intri_dir = os.path.join(path, "intrinsics")

    pose_file_list = sorted(glob(os.path.join(pose_dir, "*.txt")), 
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    intri_file_list = sorted(glob(os.path.join(intri_dir, "*.txt")), 
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    assert(len(pose_file_list) == len(intri_file_list))
    num_camera = len(pose_file_list)

    Ks = np.empty((num_camera, 3, 3), dtype=np.float32)
    C2Ws = np.empty((num_camera, 3, 4), dtype=np.float32)

    index = 0 
    for pose_file, intri_file in zip(pose_file_list, intri_file_list):
        Ks[index] = load_intrinsic_tat(intri_file)
        C2Ws[index] = load_camera_pose_tat(pose_file)
        index += 1
    print(f"\n====== Load camera Ks {Ks.shape} C2Ws {C2Ws.shape} ======\n")
    return Ks, C2Ws

# =================================== Tanks and temples Data ===================================



# if __name__ == '__main__':
#     # load_camera_pose_tat('/data/wxc/sig24/data/sig23/playground/train/pose/00002.txt')
#     # K = load_intrinsic_tat('/data/wxc/sig24/data/sig23/playground/train/intrinsics/00002.txt')
#     # print(K)
#     Ks, C2Ws = load_cameras_tat('/data/wxc/sig24/data/sig23/playground/train')