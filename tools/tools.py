import numpy as np 
import cv2, os, yaml, re
from torch._C import device, dtype 
from plyfile import PlyData, PlyElement 
from tqdm import tqdm  
import pickle 
import torch 
import imageio
import matplotlib.pyplot as plt 
import threading
import time 
from matplotlib import cm
from easydict import EasyDict as edict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 定义一个函数来绘制3D箭头
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def plot_camera(ax, position, rotation_matrix, scale=1.0, color='r'):
    # 四棱锥的顶点
    camera_body = np.array([[0, 0, 0],   # 相机中心
                            [-0.5, 0.4, 1],  # 图像平面顶点
                            [0.5, 0.4, 1],   # 图像平面顶点
                            [0.5, -0.4, 1],  # 图像平面顶点
                            [-0.5, -0.4,1]]) * scale  # 图像平面顶点
    
    # 交换z轴和y轴
    swap_matrix = np.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])

    # 旋转并平移四棱锥
    camera_body = ((rotation_matrix @ camera_body.T).T + position) @ swap_matrix.T
    position = position @ swap_matrix.T
    
    # 定义四棱锥的面
    faces = [[camera_body[0], camera_body[1], camera_body[2]],  # 面 1
             [camera_body[0], camera_body[2], camera_body[3]],  # 面 2
             [camera_body[0], camera_body[3], camera_body[4]],  # 面 3
             [camera_body[0], camera_body[4], camera_body[1]],  # 面 4
             [camera_body[1], camera_body[2], camera_body[3], camera_body[4]]]  # 图像平面

    # 绘制四棱锥的面
    poly3d = Poly3DCollection(faces[:-1], facecolors='lightgray', linewidths=1, edgecolors=color, alpha=0.25)
    ax.add_collection3d(poly3d)
    
    # 绘制图像平面
    image_plane = Poly3DCollection([faces[-1]], facecolors='lightgray', linewidths=2, edgecolors=color, alpha=0.5)
    ax.add_collection3d(image_plane)
    
    # 标记相机中心
    # ax.scatter(position[0], position[1], position[2], c=color, marker='o')



def angle2rotation(angle):

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


    rotation = Rz(angle[2] / 180 * np.pi) @ Ry(angle[1] / 180 * np.pi) @ Rx(angle[0] / 180 * np.pi)
    return rotation

def normalize(x):
    return x / np.linalg.norm(x)

def view_matrix(z_axis, up, C):
    z_axis = normalize(z_axis)
    x_axis = normalize(np.cross(up,z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.stack([x_axis,y_axis,z_axis,C],axis=-1)

def compute_C2W(z_axis, up, focus, distance):
    x_axis = np.cross(up, z_axis)
    y_axis = np.cross(z_axis, x_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(y_axis)
    z_axis = normalize(z_axis)
    C = focus - z_axis * distance 
    # 3 x 4 
    return np.stack([x_axis,y_axis,z_axis,C],axis=-1)

def compute_spiral_camera_path(up, focus, distance, num, face, zrate, rads):
    poses = []
    # z_axis = np.array([0,-np.sin(np.pi/6.),-np.cos(np.pi/6.)])
    z_axis = face
    base_c2w = compute_C2W(z_axis, up, focus, distance)
    step = 2 / num
    for theta in np.arange(0, 2, step):
        theta = theta * np.pi 
        center = np.dot(base_c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = focus - center
        C2W = compute_C2W(z, up, focus, distance)
        poses.append(C2W)
    return np.array(poses) 

# 使用内插 LLFF 
def compute_spiral_camera_path_LLFF(C2Ws, num, near, far, zrate=1.0):
    render_poses = []

    base_center = np.mean(C2Ws[:,:,3],axis=0) # 3 
    base_z_axis = np.mean(C2Ws[:,:,2],axis=0) # 3 
    up = np.mean(C2Ws[:,:,1],axis=0) # 3 
    
    
    distance =  near + (far - near) * 0.2
    focus = base_center + base_z_axis * distance
    
    base_c2w = view_matrix(base_z_axis, up, base_center)
    tt = C2Ws[:,:3,3]
    rads = np.percentile(np.abs(tt-base_center), 90, 0)
    rads = np.array(list(rads) + [1.0])
    rads[2] = 0
    
    return compute_spiral_camera_path(up=up,focus=focus ,distance=distance ,num=num, face=base_z_axis, zrate=zrate, rads=rads)


def compute_inside_camera_path(c2w1, c2w2, num):
    c2ws = []
    
    C2W1 = c2w1
    C2W2 = c2w2
    
    up = 0.5 * C2W1[:,1] + 0.5 * C2W2[:,1]
    for step in np.linspace(0,1,num):
        center = C2W1[:,3] * step + C2W2[:,3] * (1-step)
        z = C2W1[:,2] * step + C2W2[:,2] * (1-step)
        x = np.cross(up, z)
        y = np.cross(z,x)
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)
        C2W = np.stack([x,y,z,center],axis=-1)
        c2ws += [C2W]

    return np.stack(c2ws)

        
def draw_AABB(centers, sizes, marks=[], color=(200,200,200)):
    """
    centers  N x 3 [cx, cy, cz]
    """
    init_coor = np.array(
        [[1,1,-1],[1,1,1],[-1,1,1],[-1,1,-1],
            [1,-1,-1],[1,-1,1],[-1,-1,1],[-1,-1,-1]], dtype=np.float32)

    init_face = np.array(
        [[0,1,4],[1,5,4],[0,4,7],[0,7,3],[2,3,7],[2,7,6],[1,6,5],[1,2,6],
        [0,2,1],[0,3,2],[4,5,6],[4,6,7]], dtype=np.int32) + 1

    vertex = []; face = []
    count = 0
    if isinstance(color, tuple): 
        for idx in tqdm(range(len(centers))): 
            center = centers[idx]
            size = sizes[idx]
            coords = init_coor.copy()
            coords = coords * (size / 2) + center
            
            colors = np.ones_like(coords) * 0.7
            if idx in marks:
                colors *= color
            coords = np.concatenate([coords, colors], -1)
            
            vertex += [coords]
            face += [init_face + count * 8]
            count += 1
        vertex = np.concatenate(vertex, 0)
        face = np.concatenate(face, 0)
        return vertex, face
    else:
        for idx in tqdm(range(len(centers))): 
            center = centers[idx]
            size = sizes[idx]
            coords = init_coor.copy()
            coords = coords * (size / 2) + center
            
            colors = np.ones_like(coords) * color[idx]
            coords = np.concatenate([coords, colors], -1)
            
            vertex += [coords]
            face += [init_face + count * 8]
            count += 1
        vertex = np.concatenate(vertex, 0)
        face = np.concatenate(face, 0)
        return vertex, face


def write_campara(path, ks, c2ws, H, W):
    file = open(path, "w")
    count = 0
    for k, c2w in zip(ks, c2ws):
        file.write(f"{count}\n")
        file.write(f"{k[0,0]:.2f} {k[1,1]:.2f} {k[0,2]} {k[1,2]}\n")
        file.write(f"{W} {H} 0 1000\n")
        file.write(f"{c2w[0,0]:.8f} {c2w[0,1]:.8f} {c2w[0,2]:.8f} {c2w[0,3]:.8f}\n")
        file.write(f"{c2w[1,0]:.8f} {c2w[1,1]:.8f} {c2w[1,2]:.8f} {c2w[1,3]:.8f}\n")
        file.write(f"{c2w[2,0]:.8f} {c2w[2,1]:.8f} {c2w[2,2]:.8f} {c2w[2,3]:.8f}\n")
        file.write(f"0 0 0 1\n")
        count += 1
    file.close()

def write_campara_v2(path, focal, c2ws, H, W):
    file = open(path, "w")
    count = 0
    for c2w in c2ws:
        file.write(f"{count}\n")
        file.write(f"{focal} {focal} {W/2.} {H/2.}\n")
        file.write(f"{W} {H} 0 1000\n")
        file.write(f"{c2w[0,0]:.8f} {c2w[0,1]:.8f} {c2w[0,2]:.8f} {c2w[0,3]:.8f}\n")
        file.write(f"{c2w[1,0]:.8f} {c2w[1,1]:.8f} {c2w[1,2]:.8f} {c2w[1,3]:.8f}\n")
        file.write(f"{c2w[2,0]:.8f} {c2w[2,1]:.8f} {c2w[2,2]:.8f} {c2w[2,3]:.8f}\n")
        file.write(f"0 0 0 1\n")
        count += 1
    file.close()
    
def read_campara(path, return_shape=False):
    """
    read camera paras of Indoor Scene 
    format  

    index 
    fx fy cx cy
    width height near far 
    r11 r12 r13 t1
    r21 r22 r23 t2
    r31 r32 r33 t3 (camera2world)
    0   0   0   1
    """
    trans = lambda x:float(x)
    Ks = []
    C2Ws = []
    with open(path, 'r') as f:
        lines = f.readlines()
    
    for i in range(0,len(lines), 7):
        item = lines[i:i+7]
        name = item[0].strip()
        fx,fy,cx,cy = map(trans, re.split(r"\s+",item[1].strip()))
        width, height, near, far = map(trans, re.split(r"\s+",item[2].strip()))
        r11,r12,r13,t1 = map(trans, re.split(r"\s+",item[3].strip()))
        r21,r22,r23,t2 = map(trans, re.split(r"\s+",item[4].strip()))
        r31,r32,r33,t3 = map(trans, re.split(r"\s+",item[5].strip()))
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
        RT = np.array([[r11,r12,r13,t1],
                       [r21,r22,r23,t2],
                       [r31,r32,r33,t3]],dtype=np.float32)
        Ks += [K]
        C2Ws += [RT]
    Ks = np.stack(Ks, 0)
    C2Ws = np.stack(C2Ws, 0)
    print('\n=== Finish Loading camera ==')
    print(f'Ks shape: {Ks.shape}\tC2Ws shape: {C2Ws.shape}')
    if return_shape == False:
        return Ks, C2Ws
    else:
        return Ks, C2Ws, int(height), int(width)

# numpy  get rays
def get_rays_np(H, W, K, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    # 也可以理解为摄像机坐标下z=1的平面上的点
    dirs = np.stack([(i-K[0,2])/K[0,0], (j-K[1,2])/K[1,1], np.ones_like(i)], -1) 
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d   # H x W x 3

def get_rays_torch(H, W, K, c2w):
    device = K.device
    j,i = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    dirs = torch.stack([(i-K[0,2])/K[0,0], (j-K[1,2])/K[1,1], torch.ones_like(i, device=device)], -1) 
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].clone()[None,None,].repeat(H,W,1)
    return rays_o, rays_d
    # rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

def get_bbox_by_near_far(near, far, c2ws):
    
    bbox_size = far - near 
    
    center = np.mean(c2ws[:,:,3], axis=0)
    forward_axis = np.mean(c2ws[:,:,2], axis=0)
    
    bbox_center = center + bbox_size / 2. * forward_axis
    
    bbox_size = np.array([bbox_size,bbox_size,bbox_size])
    return bbox_center, bbox_size
    

def line_scatter(A, B, step=0.01, colors=(255,255,255)):
    lam = np.arange(0,1+step,step)
    lam = np.stack([lam,lam,lam],axis=-1)
    C = (1 - lam) * A + lam * B 
    colors = np.ones_like(C) * colors
    return np.concatenate([C,colors], -1) # N x 3 

def normal_scatter(pts, normals, length=1, step=0.02):
    """
    pts N x 3 
    normals N x 3 
    """
    lam = np.arange(0, 1+step, step) # T 

    out = np.empty((lam.shape[0], normals.shape[0], 6), dtype=np.float32)

    # N x 3 
    A = pts 
    B = pts + normals * length 

    out[..., :3] = (1-lam)[:, None, None] * A[None, ...] + lam[:, None, None] * B[None, ...]
    # # N x T x 3 
    # out[..., :3] = (1-lam)[None] * pts[:,None,:] + lam[None] * (pts + normals*length)[:,None,:]

    # N x T x 3 
    out[..., 3:] = np.tile((normals[None,:,:]+1)/2.*255, (out.shape[0],1,1))
    
    return out.reshape(-1,6)
    
    

def camera_scatter_colored(R, C, color, length=1, step=0.01):
    xs = line_scatter(C, C+R[0,:]*length, step, color)
    ys = line_scatter(C, C+R[1,:]*length, step, color)
    zs = line_scatter(C, C+R[2,:]*length, step, color)   
    return np.concatenate([xs,ys,zs],axis=0)

def cameras_scatter_colored(Rs, Cs, colors, length=2, step=0.01):
    scatters = []
    for R,C,color in zip(Rs, Cs, colors):
        scatters += [camera_scatter_colored(R, C, color, length, step)]
    return np.concatenate(scatters, 0)

def camera_scatter(R, C, length=1, step=0.01):
    """
    R is world2camera rotation 
    """
    xs = line_scatter(C, C+R[0,:]*length, step, (255,0,0))
    ys = line_scatter(C, C+R[1,:]*length, step, (0,255,0))
    zs = line_scatter(C, C+R[2,:]*length, step, (0,0,255))

    return np.concatenate([xs,ys,zs],axis=0)

def cameras_scatter(Rs, Cs, length=2, step=0.01):
    scatters = []
    for R,C in zip(Rs, Cs):
        scatters += [camera_scatter(R, C, length, step)]
    return np.concatenate(scatters, 0)


def points2obj(out_path, points):
    """Converts point to obj format 
    """
    f = open(out_path, 'w')
    for item in points:
        f.write('v ' + ' '.join(list(map(lambda x:str(x), item))) + '\n')
    f.close()

def mesh2obj(out_path, vertex, face, color=None):
    f = open(out_path, 'w')
    for item in vertex:
        if color:
            f.write('v ' + ' '.join(list(map(lambda x:str(x), list(item)+list(color)))) + '\n')
        else:
            f.write('v ' + ' '.join(list(map(lambda x:str(x), list(item)))) + '\n')
    for item in face:
        f.write('f ' + ' '.join(list(map(lambda x:str(x), item))) + '\n')
    f.close()


def obj2mesh(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    vertices = []
    faces = []
    for line in tqdm(lines):
        line = line.strip()
        if line=='' or line[0] == '#':
            continue 
        line = line.split(' ')

        if line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])
        elif line[0] == 'f':
            faces.append([float(line[1]), float(line[2]), float(line[3])])

    return np.array(vertices), np.array(faces) 


def generate_video(save_path, img_list, fps=20):
    # img_list = [x.astype(np.uint8) for x in img_list]
    # imageio.mimwrite(save_path, img_list, fps=fps)
    out = cv2.VideoWriter(save_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_list[0].shape[1],img_list[0].shape[0]))
    for img in img_list:
        out.write(img[...,::-1].astype(np.uint8))
    out.release()

def generate_gif(save_path, img_list, fps):
    img_list = [x.astype(np.uint8) for x in img_list]
    imageio.mimsave(save_path, img_list, fps=fps)

def save_img_list(save_path, img_list):
    img_list = [x.astype(np.uint8) for x in img_list]
    for idx, x in enumerate(img_list):
        cv2.imwrite(os.path.join(save_path, f"{idx}.png"), x[...,::-1])



def vis_camera(c2ws):

    R_base = np.array([1,0,0,0,0,-1,0,1,0]).reshape(3,3)

    cs = c2ws[:, :3, 3]

    min_v = np.min(cs)
    max_v = np.max(cs)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlim3d(min_v, max_v)
    ax.set_ylim3d(min_v, max_v)
    ax.set_zlim3d(min_v, max_v)

    length = (max_v - min_v) * 0.2
    step = length * 0.1

    for i,c2w in enumerate(c2ws):
        C = c2w[:3,3]
        R = R_base @ c2w[:3,:3].transpose()
        xs = line_scatter(C, C+R[0,:]*length, step)
        ys = line_scatter(C, C+R[1,:]*length, step)
        zs = line_scatter(C, C+R[2,:]*length, step)

        ax.plot(xs[:,0],xs[:,1],xs[:,2], color='r')
        ax.plot(ys[:,0],ys[:,1],ys[:,2], color='g')
        ax.plot(zs[:,0],zs[:,1],zs[:,2], color='b')
    ax.legend()
    plt.show()


def read_bundle(path, only_cam=False):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']

    if lines[0][0] == '#':
        lines = lines[1:]
    
    num_cameras, num_points = list(map(lambda x: int(x), lines[0].split(' ')))
    print(f"num cameras {num_cameras}\tnum points {num_points}\n")
    lines = lines[1:]

    RTs = np.empty((num_cameras, 3, 4), dtype=np.float32)
    intrinsics = np.empty((num_cameras, 3), dtype=np.float32)
    print("\nLoad camera ...")
    for i in tqdm(range(num_cameras)):

        temp_lines = lines[0:5]

        params = list(map(lambda x: [float(item) for item in x.split(' ')],  temp_lines))

        intrinsics[i] = np.array(params[0])
        R = np.array(params[1:4], dtype=np.float32).reshape(3,3)
        T = np.array(params[4:5], dtype=np.float32).reshape(3,1)
        RTs[i] = np.concatenate([R,T], 1)

        lines = lines[5:]
    
    if only_cam:
        return intrinsics, RTs
        
    pts = np.zeros((num_points, 6), dtype=np.float32)
    vis = [list() for _ in range(num_cameras)]
    print("\nLoad points ...")

    for i in tqdm(range(num_points)):
        temp_lines = lines[i*3:i*3+3]

        params = list(map(lambda x: [float(item) for item in x.split(' ')],  temp_lines))

        pts[i] = np.array(params[0] + params[1])

        num_view = int(params[2][0])

        for j in range(num_view):
            view_idx = int(params[2][1 + j*4])
            vis[view_idx].append(i)


    return intrinsics, RTs, pts, vis 

# def color_depth_torch(depth, near=None, far=None, color_type='hsv'):
#     if near == None or far == None:
#         depth = (depth - depth.min()) / (depth.max() - depth.min())
#     else:
#         depth = (depth - near) / (far - near)

#     depth = depth.repeat(1,1,3)
#     cmap = cm.get_cmap(color_type)

#     return cmap[depth]


# if __name__ == "__main__":
#     depth = torch.rand(100,100,1) 
#     depth_color = color_depth_torch(depth)




    



