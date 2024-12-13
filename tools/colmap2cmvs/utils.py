import numpy as np 

def q2proj(Q):
    a = Q[0] * np.pi / 180
    b = Q[1] * np.pi / 180
    g = Q[2] * np.pi / 180
    s1 = np.sin(a)
    s2 = np.sin(b)
    s3 = np.sin(g)
    c1 = np.cos(a)
    c2 = np.cos(b)
    c3 = np.cos(g)

    RT = np.zeros((4,4), dtype=np.float32)
    RT[0,0] = c2*c3 
    RT[0,1] = c3*s2*s1 - s3*c1 
    RT[1,0] = s3*c2 
    RT[1,1] = s3*s2*s1 + c3*c1 
    RT[1,2] = s3*s2*c1 - c3*s1
    RT[2,0] = -s2
    RT[2,1] = c2*s1 
    RT[2,2] = c1*c2  
    RT[2,3] = Q[5]
    RT[0,2] = c3*s2*c1 + s3*s1 
    RT[0,3] = Q[3]
    RT[1,3] = Q[4]
    RT[2,2] = c2*c1 
    RT[3,3] = 1.0

    return RT  

def proj2q(RT):
    Q = np.zeros((6), dtype=np.float32)
    Q[3] = RT[0,3]
    Q[4] = RT[1,3]
    Q[5] = RT[2,3]
    if RT[2,0] == 1.0:
        Q[1] = -np.pi / 2.
        Q[2] = 0
        Q[0] = np.arctan2(-RT[0,1], RT[1,1])
    else:
        if RT[2,0] == -1.0:
            Q[1] = np.pi / 2. 
            Q[2] = 0
            Q[0] = np.arctan2(RT[0,1], RT[1,1])
        else:
            Q[1] = np.arcsin(-RT[2,0])
            if np.cos(Q[1] > 0):
                s = 1.0
            else:
                s = -1.0
            Q[0] = np.arctan2(RT[2,1]*s, RT[2,2]*s)
            Q[2] = np.arctan2(RT[1,0]*s, RT[0,0]*s)
    Q[0] = Q[0] * 180 / np.pi 
    Q[1] = Q[1] * 180 / np.pi 
    Q[2] = Q[2] * 180 / np.pi 
    for i in range(3):
        if np.abs(Q[i] > 180):
            if Q[i] > 0:
                Q[i] = Q[i] - 360
            else:
                Q[i] = Q[i] + 360
    
    return Q 



def read_cmvs_camera(path, is_c2w=False):
    with open(path, 'r') as f:
        lines = f.readlines()

    params = []
    for line in lines[1:]:
        params += list(map(lambda x:float(x), line.strip().split(' ')))
    
    RT = np.array(params, dtype=np.float32).reshape(3,4)

    if is_c2w:
        R = RT[:3,:3]
        T = RT[:3,3:4]
        R = R.transpose()
        C = -R @ T 
        return np.concatenate([R,C], 1)
    else:
        return RT 


def read_cmvs_cluster(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    if lines[0] != "SKE":
        print("Not ske file!")
        return 

    num_cameras, num_clusters = list(map(lambda x: int(x),  lines[1].split(' ')))
    print(num_cameras, num_clusters)

    clusters = []
    line_index = 1
    while True:
        line_index += 1
        if line_index >= len(lines):
            break 
        line = lines[line_index]
        if line == '':
            continue 
        line = line.split(' ')
        if len(line) == 2:
            temp = list(map(lambda x: int(x), lines[line_index+1].split(' ')))
            clusters.append(temp)
        else:
            continue 
    
    assert len(clusters) == num_clusters

    return clusters, num_cameras, num_clusters