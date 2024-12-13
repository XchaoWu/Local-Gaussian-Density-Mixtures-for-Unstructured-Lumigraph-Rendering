'''
install: 
pip install -i https://mirrors.aliyun.com/pypi/simple pyqtgraph
pip install -i https://mirrors.aliyun.com/pypi/simple PyQt5
pip install pyOpenGL -i https://pypi.douban.com/simple
'''
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import tools, time 
from colmap2cmvs import utils
from matplotlib import cm
from colmap2cmvs import colmap2bundle

def vis_cluster(c2ws, pts, bboxes, camera_scale=0.1):
    '''visualize camera, point clouds and bounding boxes
       cameras are rotated from direction (1, 0, 0)

    Args:
        c2ws (K x 3 x 4 ndarray): camera to world system matrix, K 3x3 rotation matrix and K 3x1 translate matrix
        pts (N x 6 ndarray): point clouds, N x (x, y, z, R, G, B)
        bboxes (M x 6 ndarray): bounding boxes, , M x (x_min, y_min, z_min, x_max, y_max, z_max)
    '''
    # configs
    SHOWGRID = False  # whether to draw x-y grid
    SHOWAXIS = True  # whether to draw axis
    POINTSIZE = 5  # size of points in point clouds
    GRID = 25  # size of grid
    BOUNDINGCOLOR = (.5, .5, .5, .3)  # color of bounding box
    CAMERACOLOR = (1, 0, 0, 1)  # color of cameras

    # preparing the widget & grid
    app = pg.mkQApp("Visualize cluster")
    w = gl.GLViewWidget()
    w.show()
    w.resize(1000, 800)
    w.setWindowTitle('Visualize cluster')
    w.setCameraPosition(distance=50)
    if SHOWGRID:
        griditem = gl.GLGridItem(size=QtGui.QVector3D(GRID * 2, GRID * 2, 1))
        w.addItem(griditem)
    if SHOWAXIS:
        # x is blue, y is yellow, z is green
        axisitem = gl.GLAxisItem(size=QtGui.QVector3D(GRID, GRID, GRID))
        w.addItem(axisitem)

    # preparing camera
    vertexes = np.array([[0, 0, 0], [3, 1, 1], [3, 1, -1], [3, -1, -1], [3, -1, 1]]) * camera_scale
    faces = np.array([[1, 0, 2], [2, 0, 3], [3, 0, 4], [4, 0, 1]])
    for mat in c2ws:   
        rotation = mat[:, :3]   # rotation matrix
        v = np.array([np.dot(vert, rotation) for vert in vertexes])
        camera = gl.GLMeshItem(vertexes=v,
                               faces=faces,
                               drawFaces=False,
                               drawEdges=True,
                               edgeColor=CAMERACOLOR,
                               smooth=False,
                               shader='balloon',
                               glOptions='additive')
        camera.translate(mat[0][3], mat[1][3], mat[2][3])   # translate matrix
        w.addItem(camera)

    # preparing point clouds
    N = pts.shape[0]
    pos = pts[:, 0:3]
    size = np.full((N), POINTSIZE)
    color = np.hstack((pts[:, 3:], np.ones((N, 1))))
    # If True, spots are always the same size regardless of scaling, and size is given in px.
    # Otherwise, size is in scene coordinates and the spots scale with the view.
    ppts = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=True)
    w.addItem(ppts)

    # preparing bounding boxes
    vertexes = np.array([[[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min], [x_min, y_min, z_max],
                          [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]] for (x_min, y_min, z_min, x_max, y_max, z_max) in bboxes])
    faces = np.array([[0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0], [4, 7, 6], [4, 6, 5], [0, 1, 2], [0, 2, 3]])
    for vert in vertexes:
        w.addItem(gl.GLMeshItem(vertexes=vert, faces=faces, drawEdges=False, color=BOUNDINGCOLOR, smooth=False, shader='balloon', glOptions='additive'))

    # dispalay the widget
    pg.exec()
    return

def vis_boundingbox(data_dir):
    intrinsics, RTs, pts, vis = tools.read_bundle(os.path.join(data_dir, "bundle.rd.out"))
    # print(pts.shape)
    Rs = RTs[:,:3,:3]
    Ts = RTs[:,:3,3:4]
    Cs = -1 * Rs.transpose(0,2,1) @ Ts 

    center = [0,0,0]
    # boundingbox_x = 10
    # boundingbox_y = 10
    # boundingbox_z = 10
    vertex, face = tools.draw_AABB([center], [10])
    tools.mesh2obj(os.path.join(data_dir, "bbox.obj"), vertex, face)




def export_cluster_v2(data_dir):

    intrinsics, RTs, pts, vis = tools.read_bundle(os.path.join(data_dir, "bundle.rd.out"))

    clusters, num_cameras, num_clusters = utils.read_cmvs_cluster(os.path.join(data_dir, "ske.dat"))

    cmap = cm.get_cmap('hsv')

    total_points = []
    step = 1.0 / num_clusters
    seed = 0
    for idx,cluster in enumerate(clusters):
        cameras = RTs[cluster]

        point_cloud = pts.copy()
        # print(cluster)
        for k in cluster:
            for pidx in vis[k]:
                point_cloud[pidx][3] = 1.0
                point_cloud[pidx][4] = 0
                point_cloud[pidx][5] = 0
        # point_cloud = np.array(point_cloud).reshape(-1,6)

        Rs = cameras[:,:3,:3]
        Ts = cameras[:,:3,3:4]
        Cs = -1 * np.einsum("ijk,ikl->ijl", Rs.transpose(0,2,1), Ts)
        Cs = Cs.squeeze(-1)
        color = tuple(list(cmap(seed)[:3]))
        seed += step
        print(seed, color)
        colored_points = tools.cameras_scatter_colored(Rs, Cs, color, length=0.2)
        points = tools.cameras_scatter(Rs, Cs, length=0.2)

        c2ws = np.concatenate([Rs.transpose(0,2,1), Cs[...,None]], -1)
        # vis_cluster(c2ws, colored_points, [])

        tools.points2obj(os.path.join(data_dir, f"cameras_{idx}.obj"), np.concatenate([point_cloud, points], 0))

        total_points.append(colored_points)
    total_points = np.concatenate(total_points, 0)
    tools.points2obj(os.path.join(data_dir, "cameras.obj"), total_points)



if __name__ == '__main__':

    import os,sys 
    from glob import glob 
    import tools 
    from colmap2cmvs.utils import read_cmvs_camera
    # data_dir = '/home/yons/projects/cmvs/data/hall-after-bundler-cmvs/pmvs/'
    data_dir = '/data/wxc/sig24/data/sig23/playground/cmvs/pmvs'
    # colmap_dir = '/data/wxc/sig24/data/sig23/salon/'
    # data_dir = '/home/yons/projects/cmvs/data/hall-after-bundler-cmvs/pmvs/txt'

    vis_boundingbox(data_dir)
    # export_cluster_v2(data_dir)

    # export_cluster(os.path.join(data_dir, "ske.dat"),
    #                 os.path.join(data_dir, "poses"), 
    #                 os.path.join(data_dir, "perview"),
    #                 colmap_dir)

    # files = glob(os.path.join(data_dir, "*.txt"))
    # files.sort()


    # c2ws = []
    # for file in files:
    #     _, c2w = read_cmvs_camera(file, False, True)
    #     c2ws.append(c2w)
    # c2ws = np.array(c2ws)
    # print(c2ws.shape)

    # # indices = [34,196,240,243]
    # # c2ws = c2ws[indices]
    # points = tools.cameras_scatter(c2ws[:,:3,:3].transpose(0,2,1), c2ws[:,:3,3], length=0.6)
    # tools.points2obj("cameras.obj", points)
    exit()

    # N = 300  # N points in point clouds
    # M = 3  # M bounding boxes
    # # for camera matrix
    # c2ws = np.array([[[-0.5, -0.5, -0.707, 15], 
    #                   [0.707, -0.707, 0, 15], 
    #                   [-0.5, -0.5, 0.707, 15]], 
    #                  [[1, 0, 0, 10], [0, 1, 0, -10], [0, 0, 1, 20]]])
    # # for point clouds
    # pos = np.random.random((N, 3))
    # pos[:100, :] = np.random.random((100, 3)) * 3
    # pos[100:200, :1] = np.random.random((100, 1)) * (-3)
    # pos[100:200, 1:] = np.random.random((100, 2)) * 3
    # pos[200:, :] = np.random.random((100, 3)) * (-3)
    # colors = np.random.random((N, 3))
    # pts = np.hstack((pos, colors))
    # # for bounding boxes
    # # bboxes = np.array([[0, 0, 0, 15, 15, 15], [-12, 13, 14, -20, 20, 20], [-30, -30, -30, -4, -10, -6]])
    # bboxes = np.array([])
    # vis_cluster(c2ws, pts, bboxes)
