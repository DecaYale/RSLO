import numpy as np
import quaternion
from quaternion import from_float_array as a2q
from quaternion import as_float_array as q2a
from quaternion import from_rotation_matrix as m2q, as_rotation_matrix as q2m
import rslo.utils.pose_utils_np as pun
import rslo.utils.pose_utils as tch_p
import torch.nn.functional as F
import torch

bv_3d_coords = None


def cartesian2spherical(xyz):
    """
        xyz: (N,3)
        |z
        | /y
        |/____x
    """
    theta=torch.atan2(xyz[:,1:2], xyz[:,0:1]) # atan(y/x)
    phi=torch.atan(xyz[:,2:3], torch.norm(xyz[:,0:2],dim=-1, keepdim=True) ) #atan(z/sqrt(x**2+y**2))

def meanshift_gpu(input, conf=None,chunk_size=100, iter=5,bandwidth=2.5,threshold=1e-3, dist_func_type=None,kernel_reduce=False):
    """[summary]

    Arguments:
        input {BxNxK} -- [B: batchsize, N: #points, K:data dimmension]
        dist_func -- ["Euclidean", "Cosine"]

    """ 

    def gaussian(dist, sigma=2.5,conf=None):

        if conf is None:
            # return torch.exp(-0.5 * (dist / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
            return torch.exp(-0.5 * (dist / sigma)**2)
        else:
            # return torch.exp(-0.5 * (dist / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
            return torch.exp(-0.5 * (dist / sigma)**2)*conf


    # def distance_batch(a, b):
    def kernel_dist_func(a, b, type, reduce=False):
        #a: BxNxK   b: BxCxK
        if type=='Euclidean':
            if reduce:
                #a: Bx1xNxK  b: BxCx1xK
                return torch.sqrt(((a[:,None,:] - b[:,:,None]) ** 2).sum(-1, keepdim=True)) #BxCxNx1
            else:
                return torch.sqrt(((a[:,None,:] - b[:,:,None]) ** 2) ) #BxCxNxK

        elif type == 'Cosine':
            return 1-F.cosine_similarity(a[:,None,:], b[:,:,None], dim=-1 ).unsqueeze(-1) #BxCxNx1

        else:
            raise ValueError()
    def merge_dist_func(type, reduce=False):
        if type=='Euclidean' :
            return lambda x,y: ((x-y)**2).sum(dim=-1)
        elif type=='Cosine':
            return lambda x,y: 1-F.cosine_similarity(x,y, dim=-1) #BxCxN
        else:
            raise ValueError()



    B,N,K = input.shape
    # n = len(input)
    X = input.cuda()
    for _ in range(iter):
        for i in range(0, N, chunk_size):
            # import pdb 
            # pdb.set_trace()
            s = slice(i, min(N, i + chunk_size))

            weight = gaussian(kernel_dist_func(X, X[:, s], type=dist_func_type, reduce=kernel_reduce), bandwidth, conf=conf) #BxNxK,BxCxK -> BxCxN x(K or 1) 
            # weight:BxbxN, X:BxNxk
            # weighted_sum= (weight[:, :, :,None] * X[:,None,:,:]).sum(dim=2, keepdim=False) # BxCxNx1 * Bx1xNxK  -> BxCxNxK ->   BxCxK
            weighted_sum= (weight[:, :, :] * X[:,None,:,:]).sum(dim=2, keepdim=False) # BxCxNx1 * Bx1xNxK  -> BxCxNxK ->   BxCxK
            # X[:,s] = weighted_sum / weight.sum(2, keepdim=False)[:,:, None ] #BxCxK / BxCxK
            X[:,s] = weighted_sum / weight.sum(2, keepdim=False)[:,: ] #BxCxK / BxCxK
    
  
    # cluster_map, valid_clusters, valid_indices = meanshift_find_cluster_map(X, merge_dist_func(dist_func_type),threshold=threshold)
    # return X, cluster_map, valid_clusters, valid_indices  #BxNxK
    return X



def meanshift_find_cluster_map(clusters,  dist_func, threshold=1e-3):
    """[Find clusters and the corresponding indices]
    
    Arguments:
        input {[BxNxK]} -- [description]
    """

    def merge_dist_func(type, reduce=False):
        if type=='Euclidean' :
            return lambda x,y: ((x-y)**2).sum(dim=-1)**0.5
        elif type=='Cosine':
            return lambda x,y: 1-F.cosine_similarity(x,y, dim=-1) #BxCxN
        else:
            raise ValueError()

    if isinstance(dist_func, (str)):
        dist_func = merge_dist_func(dist_func)


    B,N,K = clusters.shape
    cluster_map=torch.ones([B,N,N], device=clusters.device, dtype=torch.int64)
    # cluster_map=np.ones([B,N,N], dtype=torch.int32)
    # cluster_centers= torch.ones([B,N], device=clusters.device, dtype=torch.float32)

    valid_clusters=torch.zeros(B, dtype=torch.int64)
    valid_indices=torch.zeros([B,N], dtype=torch.int64)

    # mark = torch.ones([B,N], device=clusters.device, dtype=torch.uint8 )
    mark = torch.ones([B,N], device=clusters.device, dtype=torch.bool )
    for b in range(B):
        # import pdb 
        # pdb.set_trace()
        c = clusters[b][0] 
        # print("c:", c)

        # print(clusters[b].shape, c.shape,'!!')
        dists = dist_func(clusters[b],  c)
        # import pdb 
        # pdb.set_trace()
        indices = torch.nonzero(dists<threshold) # N'x1, N'<N
        valid_indices[b,0] = len(indices)

        
        cluster_map[b, 0, :len(indices)] = indices.squeeze()
        mark[b].index_fill_(0,indices.squeeze(-1),  0 )

        non_zero=  torch.nonzero(mark[b])
        next_center_i =non_zero[0] if len(non_zero)>0 else N

        ci=1
        while next_center_i<N:
            c = clusters[b][next_center_i] 
            # print("c:", c)
            indices = torch.nonzero(dist_func(clusters[b],c)<threshold) # N'x1, N'<N
            # print(dists.max(), dists.min(), dists.median(), dists.mean())
            cluster_map[b, ci, :len(indices)] = indices.squeeze()
            valid_indices[b,ci] = len(indices)
            mark[b].index_fill_(0, indices.squeeze(-1), 0 )
            # print(mark[b])
            non_zero=  torch.nonzero(mark[b])
            next_center_i = non_zero[0] if len(non_zero)>0 else N

            ci+=1
        valid_clusters[b] = ci
    # cluster_map[b,:valid_clusters[b],:valid_indices[b,..] ] 
    return cluster_map, valid_clusters, valid_indices
        

def gen_voxel_3d_coords(tq_map,  pc_range, return_seq=False, format='BHW3'):
    '''
    x up, y left
    '''

    assert(format in ['BHW3','B3HW'])
    dtype = tq_map.dtype
    device = tq_map.device
    input_shape = tq_map.shape
    if format == 'BHW3':
        spatial_size = input_shape[1:-1]
    else:
        spatial_size = input_shape[2:]

    if len(spatial_size)==2:
        spatial_size = [1]+list(spatial_size)

    grid_size = torch.from_numpy(
        np.array(list(spatial_size[::-1]))).to(device=device, dtype=dtype)  # attention
    pc_range = torch.from_numpy(pc_range).to(device=device, dtype=dtype)

    voxel_size = (pc_range[3:]-pc_range[:3])/grid_size

    origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*(grid_size[0]
                                                            ), (pc_range[4]-0)/(pc_range[4]-pc_range[1])*(grid_size[1]),(0-pc_range[2])/(pc_range[5]-pc_range[2]) * grid_size[2] 

    assert len(input_shape) == 4

    # size_y, size_x = spatial_size
    size_z,size_y, size_x= spatial_size
    iv, jv, kv = torch.meshgrid([
        torch.arange(size_y, dtype=dtype, device=device), 
        torch.arange(size_x, dtype=dtype, device=device),  
        torch.arange(size_z, dtype=dtype, device=device )])  #y,x,z

    # move the origin to the middle
    # minus 0.5 to shift to the centre of each grid
    xv = (jv - origin_loc[0])*voxel_size[0]
    yv = (-iv + origin_loc[1])*voxel_size[1]
    # zv = torch.zeros_like(xv)
    zv = (kv-origin_loc[2])*voxel_size[2]

    # xyzv = torch.stack([xv, yv, zv], dim=0).reshape([-1, 3])  # Nx3
    xyzv = torch.stack([xv, yv, zv], dim=-1).reshape([-1, 3])  # Nx3 #  Nx3 # fixed on 7/11/2019

    xyzv = torch.cat([xyzv]*input_shape[0], dim=0)

    if not return_seq:
        xyzv = xyzv.view(input_shape[0], spatial_size[1], spatial_size[2], spatial_size[0], 3) #b,y,x,z,c

        if format == 'B3HW':
            xyzv = xyzv.permute([0, 4, 3, 1,2]).contiguous() #b,y,x,z,c->b,c,z,y,x
            if xyzv.shape[2]==1:
                xyzv = xyzv.squeeze(2)
        else:
            xyzv = xyzv.permute([0, 3, 1, 2,4]).contiguous() #b,y,x,z,c->b,z,y,x,c
            if xyzv.shape[1]==1:
                xyzv = xyzv.squeeze(1)

    return xyzv
def set_bv_3d_coords(tq_map,  pc_range):
    '''
    x up, y left
    '''

    dtype = tq_map.dtype
    device = tq_map.device
    input_shape = tq_map.shape
    spatial_size = input_shape[2:]

    grid_size = torch.from_numpy(
        np.array(list(spatial_size[::-1]))).to(device=device, dtype=dtype)  # attention
    # voxel_size = np.array(voxel_generator.voxel_size)*4  # x,y,z
    pc_range = torch.from_numpy(pc_range).to(device=device, dtype=dtype)

    voxel_size = (pc_range[3:5]-pc_range[0:2])/grid_size

    # origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*grid_size[0], (pc_range[4]-(
    #     pc_range[4]+pc_range[1])/2)/(pc_range[4]-pc_range[1])*grid_size[1], 0
    # #fixed on 7/11/2019 the left-top of a grid is the anchor point
    origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*(grid_size[0]
                                                            ), (pc_range[4]-0)/(pc_range[4]-pc_range[1])*(grid_size[1]), 0

    assert len(input_shape) == 4

    # tq_map = tq_map.permute([0,2,3,1]).contiguous()
    # tq_map = tq_map.view(-1,7)

    # t_l = tq_map[:,:3]
    # q_l = tq_map[:,3:]

    size_y, size_x = spatial_size
    iv, jv = torch.meshgrid([torch.arange(size_y, dtype=dtype, device=device), torch.arange(
        size_x, dtype=dtype, device=device)])  # (size_x,size_y)
    # move the origin to the middle
    # minus 0.5 to shift to the centre of each grid
    # xv = (jv - origin_loc[0]+0.5)*voxel_size[0]
    # yv = (-iv + origin_loc[1]-0.5)*voxel_size[1]
    xv = (jv - origin_loc[0])*voxel_size[0]
    yv = (-iv + origin_loc[1])*voxel_size[1]

    zv = torch.zeros_like(xv)
    # xyzv = torch.stack([xv, yv, zv], dim=0).reshape([-1, 3])  # Nx3
    xyzv = torch.stack([xv, yv, zv], dim=-1).reshape([-1, 3]
                                                     )  # Nx3 #  Nx3 # fixed on 7/11/2019

    xyzv = torch.cat([xyzv]*input_shape[0], dim=0)

    global bv_3d_coords
    bv_3d_coords = xyzv
    # print(xyzv[:,0].max(), xyzv[:,1].max(), '!!!!')


def bv_3d_coords_to_bv_img_coors(bv_3d_coords, spatial_size, pc_range, normalize=False):
    device = bv_3d_coords.device
    dtype = bv_3d_coords.dtype

    size_y, size_x = spatial_size

    grid_size = torch.from_numpy(
        np.array(list(spatial_size[::-1]))).to(device=device, dtype=dtype)  # attention
    # voxel_size = np.array(voxel_generator.voxel_size)*4  # x,y,z
    pc_range = torch.from_numpy(pc_range).to(device=device, dtype=dtype)

    voxel_size = (pc_range[3:5]-pc_range[0:2])/grid_size
    # origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*grid_size[0], (pc_range[4]-(
    #     pc_range[4]+pc_range[1])/2)/(pc_range[4]-pc_range[1])*grid_size[1], 0
    # #fixed on 7/11/2019 the left-top of a grid is the anchor point
    origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*(grid_size[0]
                                                            ), (pc_range[4]-0)/(pc_range[4]-pc_range[1])*(grid_size[1]), 0

    xv, yv = bv_3d_coords[:, 0:1], bv_3d_coords[:, 1:2]

    # xv = (jv - origin_loc[0])*voxel_size[0]
    # yv = (-iv + origin_loc[1])*voxel_size[1]

    jv = xv/voxel_size[0]+origin_loc[0]
    iv = -(yv/voxel_size[1]-origin_loc[1])

    if normalize:
        # normalized to [-1,1)
        # jv = (jv - origin_loc[0])/size_x
        # iv = (iv - origin_loc[1])/size_y
        jv = 2*(jv)/(size_x) - 1
        iv = 2*(iv)/(size_y) - 1

    jiv = torch.cat([jv, iv], dim=-1)

    return jiv


def inverse_warp(features1, features2, tq_map, pc_range, padding_mode='zeros', mode='bilinear'):
    """[Warp feature2 to feature 1]

    Arguments:
        features1 {[type]} -- [description]
        features2 {[type]} -- [description]
        tq_map_gt {[type]} -- [description]
        pc_range {[type]} -- [description]
    """
    # assert tq_format in ['rx+t', 'r(x+t)']
    assert(features1.shape[2:] == features2.shape[2:])
    assert tq_map.shape[2] / \
        features1.shape[2] == tq_map.shape[3]/features1.shape[3]

    if (tq_map.shape[2] != features1.shape[2]):
        tq_map = F.interpolate(
            tq_map, scale_factor=features1.shape[2]/tq_map.shape[2], mode=mode)
    input_shape = tq_map.shape

    global bv_3d_coords
    if bv_3d_coords is None or len(bv_3d_coords) != input_shape[0]*input_shape[1]*input_shape[2]:
        set_bv_3d_coords(tq_map, pc_range)
    xyzv = bv_3d_coords

    tq_map = tq_map.permute([0, 2, 3, 1]).contiguous()
    tq_map = tq_map.view(-1, 7)

    t_l = tq_map[:, :3]
    q_l = tq_map[:, 3:]

    t_g = tch_p.rotate_vec_by_q(t=(t_l-xyzv), q=q_l) + xyzv
    t_map_g = t_g.view(input_shape[0], input_shape[2], input_shape[3], 3)
    # torch.ones([spatial_size[0], spatial_size[1], 4], np.float32)*q_g
    q_map_g = q_l.view(input_shape[0], input_shape[2], input_shape[3], 4)
    # .permute([0, 3, 1, 2]).contiguous()
    tq_map_g = torch.cat([t_map_g, q_map_g], dim=-1)

    coors2 = tch_p.rotate_vec_by_q(t=xyzv-t_g, q=tch_p.qinv(q_l))
    coors2 = bv_3d_coords_to_bv_img_coors(
        coors2, spatial_size=features1.shape[2:], pc_range=pc_range, normalize=True)

    coors2 = coors2[:, :2].reshape(
        [input_shape[0], input_shape[2], input_shape[3], 2])

    # import pdb
    # pdb.set_trace()
    # # for test
    # coors1 = bv_3d_coords_to_bv_img_coors(
    #     xyzv, spatial_size=features1.shape[2:], pc_range=pc_range, normalize=True)
    # coors1 = coors1[:, :2].reshape(
    #     [input_shape[0], input_shape[2], input_shape[3], 2])
    # valid_points = coors1.abs().max(dim=-1)[0] <= 1
    # print(coors1.max(), coors1.min(), valid_points.sum(), '!!')
    # # \for test

    warped_features2 = F.grid_sample(
        features2, coors2, padding_mode=padding_mode)

    valid_points = (coors2.abs().max(dim=-1, keepdim=True)[0] <= 1)

    valid_points = valid_points.permute(0,3,1,2 )#view(valid_points.shape[0], 1, *valid_points.shape[1:])
    # print(coors2.max(), coors2.min(), valid_points.sum(), '!!!')

    return warped_features2, valid_points


def odom_to_abs_pose(odoms):
    """[Compute the corressponding absolute poses of a sequence of odometries ]

    Arguments:
        odoms {[np.array]} -- [(N,7)]
    """

    # translation, rotation = odom_vectors[:, :3], odom_vectors[:, 3:]

    # t_prev = translation[0:1]
    # r_prev = rotation[0:1]
    # odoms = odoms

    t_prev, r_prev = odoms[0][:3].reshape([1, 3]), odoms[0][3:].reshape([1, 4])

    abs_poses = [np.array([0,0,0,1,0,0,0]).reshape(-1,7) ]
    for i in range(1, len(odoms)):

        t_cur, r_cur = odoms[i][:3].reshape(
            (1, 3)), odoms[i][3:].reshape((1, 4))

        r_cur = pun.qmult(r_prev, r_cur)
        t_cur = t_prev + \
            pun.rotate_vec_by_q(
                t_cur, r_prev)

        t_prev = t_cur
        r_prev = r_cur
        abs_poses.append(np.concatenate([t_prev, r_prev], axis=-1))

    return np.concatenate(abs_poses, axis=0)


def RT_to_tq(RT):
    """[Compute quaternion+translation vector form 3x4 RT matrix]

    Arguments:
        RT {[np.array]} -- [3x4 matrix]
    """

    assert RT.shape in [(3, 4), (4, 4)]

    q = q2a(m2q(RT[:3, :3]))
    q *= np.sign(q[0])  # constrain to hemisphere

    t = RT[:3, 3]

    # return np.concatenate([q, t])
    return np.concatenate([t, q]).reshape((-1, 7))


def tq_to_RT(tq, expand=False):
    """[Compute quaternion+translation vector form 3x4 RT matrix]

    Arguments:
        RT {[np.array]} -- [3x4 matrix]
    """

    # assert RT.shape in [(3, 4), (4, 4)]
    assert(tq.shape == (7,))
    RT = np.zeros((3, 4))

    t, q = tq[:3], tq[3:]

    RT[:3, :3] = q2m(a2q(q))
    RT[:3, 3:] = t.reshape((3, 1))
    if expand:
        RT = expand_rigid_transformation(RT)

    return RT

# # def rotate(q, point):
# def camera_to_lidar(points, r_rect, velo2cam):
#     points_shape = list(points.shape[0:-1])
#     if points.shape[-1] == 3:
#         points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
#     lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
#     return lidar_points[..., :3]


# def lidar_to_camera(points, r_rect, velo2cam):
#     points_shape = list(points.shape[:-1])
#     if points.shape[-1] == 3:
#         points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
#     camera_points = points @ (r_rect @ velo2cam).T
#     return camera_points[..., :3]


def expand_rigid_transformation(trans_matrix):
    if trans_matrix.shape == (4, 4):
        return trans_matrix
    elif trans_matrix.shape == (3, 4):
        res = np.zeros((4, 4))
        res[3, 3] = 1
        res[:3, :4] = trans_matrix
        return res
    else:
        raise ValueError(
            f"The matrix of shape {trans_matrix.shape} is not allowed!")


def cam_pose_to_lidar(cam_pose, velo_to_cam):
    """[Transform the pose in camera system to the lidar system ]
    The definition of pose: if X_0 = T_c@X_c, where X_0 denotes the point in frame 0, X_c the current frame, then T_c defines the current pose
    Arguments:
        cam_pose {[np.array]} -- [3x4 or 4x4]
        velo_to_cam {[np.array]} -- [3x4 or 4x4]
    """
    cam_pose = expand_rigid_transformation(cam_pose)
    velo_to_cam = expand_rigid_transformation(velo_to_cam)

    # cam_to_lidar = cam_pose@velo_to_cam
    cam_to_lidar = np.linalg.inv(velo_to_cam)@ cam_pose@ velo_to_cam

    return cam_to_lidar


def compute_rel_pose(pose1, pose2):
    """[Compute the relative transformation from two poses, \ie, the transformation of coordinates in system 2 to 1]

    Arguments:
        pose1 {[np.array]} -- [(7,) translation+rotation]
        pose2 {[np.array]} -- [(7,) translation+rotation]
    """
    q1, t1 = a2q(pose1[3:]), np.quaternion(0, *[x for x in pose1[:3]])
    q2, t2 = a2q(pose2[3:]), np.quaternion(0, *[x for x in pose2[:3]])

    q = q1.conjugate()*q2
    t = q1.conjugate()*(t2-t1)*q1
    # print(q, t, np.concatenate([q2a(q), q2a(t)[1:]]), '!!!')
    return np.concatenate([q2a(q), q2a(t)[1:]])
