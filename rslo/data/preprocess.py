# import open3d as o3d
import pathlib
import pickle
import time
from collections import defaultdict, OrderedDict
from functools import partial

import cv2
import numpy as np
import quaternion
from skimage import io as imgio
from rslo.utils.geometric import RT_to_tq, compute_rel_pose
# from rslo.core import box_np_ops
# from rslo.core import preprocess as prep
# from rslo.core.geometry import points_in_convex_polygon_3d_jit
from rslo.data import kitti_common as kitti
# from rslo.utils import simplevis
from rslo.utils.timer import simple_timer

import seaborn as sns
import matplotlib.pyplot as plt
from collections.abc import Iterable
# from rslo.data.kitti_dataset import generate_pointwise_local_transformation,generate_pointwise_local_transformation_3d,generate_pointwise_local_transformation_tch
import torch
import torch.nn.functional as F
from rslo.utils.pose_utils_np import *
import quaternion



def generate_cyc_vo(pose_seq):
    assert len(pose_seq) > 1
    seq_len = len(pose_seq)

    vos = []
    for i in range(0, seq_len):
        for j in range(i+1, seq_len):
            # vos.append(calc_vo(pose_seq[i], pose_seq[j]).squeeze(axis=0))
            vo = calc_vo(pose_seq[i], pose_seq[j]).squeeze(axis=0)
            vo[3:] *= np.sign(vo[3])
            vos.append(vo)
    vos = np.stack(vos, axis=0)  # (7*(seq_len(seq_len-1)/2),)
    return vos


def merge_second_batch(batch_list):
    # [batch][key][seq]->example[key][seq][batch]
    example_merged = defaultdict(list)
    for example in batch_list:  # batch dim
        for k, v in example.items():  # key dim
            # assert isinstance(v, list)
            if isinstance(v, list):

                seq_len = len(v)
                if k not in example_merged:
                    example_merged[k] = [[] for i in range(seq_len)]
                for i, vi in enumerate(v):  # seq dim
                    example_merged[k][i].append(vi)

            else:
                example_merged[k].append(v)

    ret = {}
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points'
        ]:
            if isinstance(elems[0], list):
                ret[key] = []
                for e in elems:
                    ret[key].append(np.concatenate(e, axis=0))
            else:
                ret[key] = np.concatenate(elems, axis=0)

        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):  # different seq
                assert isinstance(
                    coor, list), f'{type(coor[0])} is not allowed!'
                coor_pad = []
                for j, _ in enumerate(coor):  # different batch
                    # coor_pad = np.pad(
                    #     coor[j], ((0, 0), (1, 0)), mode='constant', constant_values=i) # ! i is batch index
                    coor_pad.append(np.pad(
                        coor[j], ((0, 0), (1, 0)), mode='constant', constant_values=j))  # ! j is batch index
                # coors.append(coor_pad)
                coors.append(np.concatenate(coor_pad, axis=0))
            # ret[key] = np.concatenate(coors, axis=0)
            ret[key] = coors
        elif key == 'tq_maps':
            # ret[key] = torch.stack(elems, dim=0)
            ret[key] = []
            for e in elems:
                ret[key].append(torch.stack(e, dim=0))
        elif key in ['odometry', "icp_odometry"]:
            ret[key] = np.stack(elems, axis=0)
        elif key in ['lidar_seqs', "normal_gt_seqs"]:
            # ret[key] = elems
            min_len = elems[0][0].shape[0]
            for t in range(len(elems)):
                for b in range(len(elems[t]) ):
                    min_len = min(min_len, elems[t][b].shape[0] )
            for t in range(len(elems)):
                for b in range(len(elems[t])):
                    elems[t][b] = elems[t][b][ np.random.choice(len(elems[t][b]),min_len, replace=False )] 
                elems[t] = np.stack(elems[t], axis=0)
            ret[key]=elems
        elif key in ['hier_points']:
            hier_min_lens = []
            for h in range(len(elems[0][0]) ):
                min_len = elems[0][0][h].shape[0]
                for t in range(len(elems)):
                    for b in range(len(elems[t]) ):
                        min_len = min(min_len, elems[t][b][h].shape[0] )
                hier_min_lens.append(min_len) 
            new_elems=[]
            for t in range(len(elems)):
                for b in range(len(elems[t])):
                    for h in range(len(elems[t][b])):
                        elems[t][b][h] = elems[t][b][h][ np.random.choice(len(elems[t][b][h]), hier_min_lens[h], replace=False ) ] 
                # elems[t] = np.stack(elems[t], axis=0)
                new_elems.append([])
                for h in range(len(elems[0][0] )):
                    new_elems[t].append( 
                        np.stack([elems[t][b][h] for b in  range(len(elems[t])) ], axis=0)  )
                
            ret[key]=new_elems
        elif key == 'metrics':
            ret[key] = elems
        else:
            ret[key] = []
            for e in elems:
                ret[key].append(np.stack(e, axis=0))
    return ret


def merge_second_batch_multigpu(batch_list):
    # [batch][key][time]-> example_merged[key][time][batch]
    example_merged = defaultdict(list)
    for example in batch_list:  # batch
        for k, v in example.items():  # key
            # assert isinstance(v, list), f"{type(v)} is not allowed!"
            if isinstance(v, list):
                seq_len = len(v)
                if k not in example_merged:
                    example_merged[k] = [[] for i in range(seq_len)]
                for i, vi in enumerate(v):  # seq
                    example_merged[k][i].append(vi)
            else:
                example_merged[k].append(v)

            # example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():

        if key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):  # seq
                assert isinstance(
                    coor, list), f"{type(coor[0])} is not allowed!"
                coor_pad = []
                for j, _ in enumerate(coor):  # batch
                    coor_pad.append(np.pad(
                        coor[j], ((0, 0), (1, 0)), mode='constant', constant_values=j))
                coors.append(np.stack(coor_pad, axis=0))

            # ret[key] = np.stack(coors, axis=0)
            ret[key] = coors

        elif key in ['gt_names', 'gt_classes', 'gt_boxes']:
            continue
        elif key in ['tq_maps']:
            ret[key] = torch.stack(elems, dim=0)
        elif key in ['odometry']:
            ret[key] = np.stack(elems, axis=0)
        else:
            seqs = []
            for i, elem in enumerate(elems):  # seq
                seqs.append(np.stack(elem, axis=0))

            # ret[key] = np.stack(elems, axis=0)
            ret[key] = seqs
    return ret


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def estimate_normal(points, radius=0.5, max_nn=30, camera_location=(0,0,0) ):
    '''
        points: Nx3 
        down_sample_voxel_size: float scalar
    '''
    
    if isinstance(points, np.ndarray ):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif isinstance(points, o3d.geometry.PointCloud ):
        pcd = points
    else:
        raise NotImplementedError
    # pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30) )
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30) )
    
    pcd.orient_normals_towards_camera_location(camera_location)
    return pcd
def down_sample_points(points, down_sample_voxel_size ):
    '''
        points: Nx3 
        down_sample_voxel_size: float scalar
    '''

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif isinstance(points, o3d.geometry.PointCloud ):
        pcd = points
    else:
        raise NotImplementedError

    pcd = pcd.voxel_down_sample(down_sample_voxel_size)

    return pcd


def flip_odometry(old_odom):
    # old_odom = input_dict['odometry'][odom_idx]
    q_old = np.quaternion(*list(old_odom[3:]))
    r_old = quaternion.as_rotation_matrix(q_old)
    t_old = old_odom[:3]
    f = np.array([1, 0, 0,
                    0, -1, 0,
                    0, 0, 1]).reshape([3, 3])
    r_new = f@r_old@f.T
    t_new = f@t_old
    q_new = quaternion.from_rotation_matrix(r_new)
    q_new = quaternion.as_float_array(q_new)
    q_new *= np.sign(q_new[0])
    new_odom = np.concatenate(
        [t_new, q_new], axis=-1)
    return new_odom
def prep_pointcloud(input_dict,
                    root_path,
                    voxel_generator,
                    max_voxels=20000,
                    training=True,
                    shuffle_points=False,
                    num_point_features=4,
                    anchor_area_threshold=1,
                    random_crop=False,
                    multi_gpu=False,
                    use_dist=False,
                    min_points_in_gt=-1,
                    random_flip_x=False,
                    random_flip_y=False,
                    sample_importance=1.0,
                    rand_rotation_eps=0,
                    rand_translation_eps=0,
                    rand_aug_ratio=0,
                    gen_tq_map=False,
                    do_pre_transform=False,
                    out_dtype=np.float32,
                    seed=None,
                    cubic_tq_map=False,
                    downsample_voxel_sizes=None,
                    ):
    """convert point cloud to voxels, create targets if ground truths
    exists.

    input_dict format: dataset.get_sensor_data format

    """
    t = time.time()
    if seed is not None:
        np.random.seed(seed)
    # class_names = target_assigner.classes
    # points = input_dict["lidar"]["points"]
    """
      input_dict = {
           "seq_idx": [],
            "frame_idx": [],
            "lidar_seq": [],
            "cam_pose_seq": [],
        }
    """
    points_list = input_dict["lidar_seq"]
    # normal_gt_list=input_dict["normal_gt_seq"]

    if not isinstance(points_list, Iterable):
        points_list = [points_list]

    hier_points=input_dict.get("hier_points_seq", None)
    if hier_points is None:
        # print("Warning: key 'hier_points_seq' has not been found in the input dict")
        hier_points = [ [np.concatenate([p[:,:3], p[:,4:7]], axis=-1) ] for p in points_list]

    metrics = {}

    if training:
        """
        boxes_lidar = gt_dict["gt_boxes"]
        bev_map = simplevis.nuscene_vis(points, boxes_lidar)
        cv2.imshow('pre-noise', bev_map)
        """

        # if min_points_in_gt > 0:
        #     # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
        #     point_counts = box_np_ops.points_count_rbbox(
        #         points, gt_dict["gt_boxes"])
        #     mask = point_counts >= min_points_in_gt
        #     _dict_select(gt_dict, mask)
        # gt_boxes_mask = np.array(
        #     [n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_)

        pc_range = voxel_generator.point_cloud_range
        # group_ids = None
        # if "group_ids" in gt_dict:
        #     group_ids = gt_dict["group_ids"]

        # should remove unrelated objects after noise per object
        # for k, v in gt_dict.items():
        #     print(k, v.shape)

        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]

    if 0:  # shuffle_points:
        # shuffle is a little slow.
        # np.random.shuffle(points)
        for points in points_list:
            np.random.shuffle(points)
    if random_flip_y and np.random.rand() > 0.5:
        #flip lidar points
        for i, points in enumerate(points_list):
            points_list[i][:, 1] = -points_list[i][:, 1]  # flip y
            # if len(points_list[i][0]) >= 8:
            if len(points_list[i][0]) >= 6:
                points_list[i][:, 5] = -points_list[i][:, 5]  # flip normal_y
            
            if len(points_list[i][0]) >= 9:
                points_list[i][:, 8] = -points_list[i][:, 8]  # flip normal_y
        #flip normal_gt (if exists)
        # for i,normal in enumerate(normal_gt_list):
        #     normal_gt_list[i][:,1]=-normal_gt_list[i][:,1] #flip normal_y

        #flip hier lidar points 
        for i, hier_point in enumerate(hier_points):
            for h, _ in enumerate(hier_point):
                hier_points[i][h][:,1] = - hier_points[i][h][:,1] #flip y 
                hier_points[i][h][:,4] = - hier_points[i][h][:,4] #flip normal_y 


        seq_len = len(input_dict["frame_idx"])
        odom_idx = 0
        for i in range(0, seq_len):
            for j in range(i+1, seq_len):
                # [odom_idx:odom_idx+1]
                old_odom = input_dict['odometry'][odom_idx]
                # q_old = np.quaternion(*list(old_odom[3:]))
                # # print(q_old,'!!1')
                # r_old = quaternion.as_rotation_matrix(q_old)
                # t_old = old_odom[:3]

                # f = np.array([1, 0, 0,
                #               0, -1, 0,
                #               0, 0, 1]).reshape([3, 3])
                # r_new = f@r_old@f.T
                # t_new = f@t_old

                # q_new = quaternion.from_rotation_matrix(r_new)
                # q_new = quaternion.as_float_array(q_new)
                # q_new *= np.sign(q_new[0])
                # # print(t_new, q_new, '!!!')
                # input_dict['odometry'][odom_idx] = np.concatenate(
                #     [t_new, q_new], axis=-1)
                new_odom = flip_odometry(old_odom)
                input_dict['odometry'][odom_idx] = new_odom

                if input_dict.get('icp_odometry', None) is not None:
                    old_icp = input_dict['icp_odometry'][odom_idx]
                    input_dict['icp_odometry'][odom_idx] = flip_odometry(old_icp)

                odom_idx += 1


    if rand_aug_ratio > 0:
        # rot_vecs = [quaternion.as_rotation_vector(odom) for odom in odoms]
        seq_len = len(input_dict["frame_idx"])
        aug_ratio = np.random.uniform(-rand_aug_ratio, rand_aug_ratio, seq_len)

        abs_poses = [np.array([0, 0, 0, 1, 0, 0, 0]).reshape([-1, 7])]
        for i in range(seq_len-1):
            abs_poses.append(input_dict['odometry'][i:i+1])

        new_abs_poses = []
        for i in range(seq_len):
            if i==0: #do not augment the first one 
                new_abs_poses.append(abs_poses[i])
                continue
            if i+1 < seq_len:
                # new_pose = abs_poses[i]+(abs_poses[i+1]-abs_poses[i])*aug_ratio[i]
                # new_pose = abs_poses[i] + \
                #     (abs_poses[i+1]-abs_poses[i])*aug_ratio[i]
                new_pose = abs_poses[i] + \
                    (abs_poses[i+1]-abs_poses[i])*aug_ratio[i]
                new_pose_q = quaternion.slerp(quaternion.from_float_array(
                    abs_poses[i][:, 3:]), quaternion.from_float_array(abs_poses[i+1][:, 3:]), 0, 1, aug_ratio[i])

                new_pose[:, 3:] = quaternion.as_float_array(new_pose_q)
                # print((new_pose-new_pose_bak)/(new_pose_bak+1e-12),'!!2')

            else:
                # new_pose = abs_poses[i] + \
                #     (abs_poses[i]-abs_poses[i-1])*aug_ratio[i]
                new_pose = abs_poses[i] + \
                    (abs_poses[i-1]-abs_poses[i])*(-aug_ratio[i])
                # print(new_pose,'1')
                new_pose_q = quaternion.slerp(quaternion.from_float_array(
                    abs_poses[i][:, 3:]), quaternion.from_float_array(abs_poses[i-1][:, 3:]), 0, 1, -aug_ratio[i])
                new_pose[:, 3:] = quaternion.as_float_array(new_pose_q)
                # print(new_pose,'2')

            new_pose[:, 3:] = normalize(new_pose[:, 3:], dim=1)
            new_pose[:, 3:] *= np.sign(new_pose[:, 3])
            # print(new_pose)
            new_abs_poses.append(new_pose)

        new_odoms = []

        input_dict['odometry'] = generate_cyc_vo(new_abs_poses)

        # apply the transformation
        for i, points in enumerate(points_list):
            transformation = calc_vo(new_abs_poses[i], abs_poses[i])

            points_list[i][:, :3] = rotate_vec_by_q(
                points[:, :3], transformation[:, 3:]) + transformation[:, :3]
            if len(points_list[i][0]) >= 7:
                # transform normals
                points_list[i][:, 4:7] = rotate_vec_by_q(
                    points[:, 4:7], transformation[:, 3:])  # + delta_ts[i:i+1]
            #TODO
            # if len(normal_gt_list)>0:
            if len(points_list[i][0]) >= 9:
                points_list[i][:, 7:]=rotate_vec_by_q(
                    points_list[i][:, 7:], transformation[:, 3:]) 
        #transform the hier lidar points 
        for i, hier_point in enumerate(hier_points):
            transformation = calc_vo(new_abs_poses[i], abs_poses[i])
            for h, _ in enumerate(hier_point):
                hier_points[i][h][:,:3] =  rotate_vec_by_q(
                hier_points[i][h][:,:3], transformation[:, 3:]) + transformation[:, :3]
                # transform normals
                hier_points[i][h][:,3:6] = rotate_vec_by_q(
                    hier_points[i][h][:,3:6], transformation[:, 3:]) 

        # [0, -40, -3, 70.4, 40, 1]
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size

    # print(voxel_size, pc_range, grid_size,'!!')
    # [352, 400]
    t1 = time.time()

    voxels = []
    coordinates = []
    num_points = []
    num_voxels = []
    frame_inds = []
    seq_inds = []
    coor_to_voxelidx=[]
    if not multi_gpu or use_dist:
        for i, points in enumerate(points_list):
            # if do_pre_transform>0:
            #     points = points[points[:,-1]<do_pre_transform][:,:8] # remove the dynamic objects

                # print(points.shape, flush=True)
            # # if i >= 0 and do_pre_transform>0:
            # if i >= 1 and do_pre_transform>0:
            #     # only for testing purpose
            #     assert(len(input_dict['odometry']) == 1)
            #     t_ = input_dict['odometry'][0:1, :3]
            #     q_ = input_dict['odometry'][0:1, 3:]
            #     trans_xyz = rotate_vec_by_q(
            #         points[:, :3], t_) + t_
            #     # trans_xyz = points[:,:3]
            #     # trans_xyz[:,:2]=  150/(np.linalg.norm(points[:,:2], axis=1, keepdims=True)+0.1) **2 * trans_xyz[:,:2]
            #     points = np.concatenate([trans_xyz, points], axis=1)
            res = voxel_generator.generate(points, max_voxels)
            
            # if i >= 1 and do_pre_transform>0:
            #     # only for testing purpose
            #     res["voxels"] = res["voxels"][:, :, 3:]

            voxels.append(res["voxels"])
            # print(res["voxels"].shape, input_dict['velodyne_path'])
            coordinates.append(res["coordinates"])
            num_points.append(res["num_points_per_voxel"])
            num_voxels.append(np.array([voxels[-1].shape[0]], dtype=np.int64))
            # coor_to_voxelidx.append(voxel_generator._coor_to_voxelidx)
    else:
        for points in points_list:
            res = voxel_generator.generate_multi_gpu(
                points, max_voxels)
            voxels.append(res["voxels"])
            coordinates.append(res["coordinates"])
            num_points.append(res["num_points_per_voxel"])
            num_voxels.append(np.array([res["voxel_num"]], dtype=np.int64))

    ########################################## down sample lidar point ##########################################
    # hier_points = OrderedDict() 
    # origin_pcds = [estimate_normal(points[:,:3]) for points in points_list  ]
    ''' 
    origin_pcds = []
    for points in points_list:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        if points.shape[-1]>4:
            pcd.normals = o3d.utility.Vector3dVector(points[:,4:7])
        else:
            pcd = estimate_normal(points[:,:3])
        origin_pcds.append(pcd)
        

    # origin_points = [np.concatenate([np.asarray(pcd.points), np.asarray(pcd.normals)], axis=-1) for pcd in origin_pcds ] 
    # origin_points = [np.asarray(pcd.points) for pcd in origin_pcds ] 
    # hier_points = [origin_points] 
    hier_points = [] # sequence of hiearachical points
    for origin_pcd in origin_pcds:
        hier_points_list = []
        for i, ds in enumerate(downsample_voxel_sizes):
            # print(ds,flush=True)
            # for origin_pcd in origin_pcds:
            # for origin_pcd in hier_points[-1]:
            pcd = down_sample_points(origin_pcd, down_sample_voxel_size=ds) 
            hier_points_list.append(np.concatenate([np.asarray(pcd.points), np.asarray(pcd.normals)], axis=-1) )
                # new_points_list.append(np.asarray(pcd.points) )
        hier_points.append(hier_points_list)
    '''
    # hier_points = [[p[:len(p)//2,:6]] for p in points_list]
    # hier_points_list =  input_dict["hier_points_seq"]
    # hier_points=input_dict.get("hier_points_seq", None)
    # # for i,_ in enumerate(hier_points):
    # #     hier_points[i] = [np.concatenate([points_list[i][:,:3], points_list[i][:,4:7]], axis=-1 ) ] + hier_points[i]
    # if hier_points is None:
    #     # print("Warning: key 'hier_points_seq' has not been found in the input dict")
    #     hier_points = [ [np.concatenate([p[:,:3], p[:,4:7]], axis=-1) ] for p in points_list]
    

            

        
        



    ########################################## \down sample lidar point #########################################
    
    if 0: #gen_tq_map:
        
        grid_size = np.array(voxel_generator.grid_size)//4
        # voxel_size = np.array(voxel_generator.voxel_size)*4  # x,y,z
        voxel_size = (np.array(pc_range[3:])-np.array(pc_range[0:3]))/grid_size
        pc_range = voxel_generator.point_cloud_range
        if cubic_tq_map:
            spatial_size = grid_size
            tq_map_generator = generate_pointwise_local_transformation_3d
        else:
            spatial_size = grid_size[:2]
            tq_map_generator = generate_pointwise_local_transformation
        # origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*grid_size[0], (pc_range[4]-(
        #     pc_range[4]+pc_range[1])/2)/(pc_range[4]-pc_range[1])*grid_size[1], 0
        # # # fixed on 7/11/2019
        # origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*(grid_size[0]
        #                                                         ), (pc_range[4]-0)/(pc_range[4]-pc_range[1])*(grid_size[1]), 0
        # the location of camera/lidar in voxel/bv coords
        origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*(grid_size[0]
                                                                ), (pc_range[4]-0)/(pc_range[4]-pc_range[1])*(grid_size[1]), (0-pc_range[2])/(pc_range[5]-pc_range[2]) * grid_size[2]

        tq_maps = []
        for tq in input_dict['odometry']:
            tq_maps.append(tq_map_generator(
                tq,
                # spatial_size=grid_size[:2],
                spatial_size=spatial_size,
                # origin_loc=np.array([0, grid_size[1]/2, 0]),
                origin_loc=origin_loc,  # np.array(origin_loc),
                voxel_size=voxel_size,
                inv_trans_factor=-1 if not do_pre_transform else 150
            ))
        # tq_maps = np.stack(tq_maps, axis=0)
        tq_maps = torch.stack(tq_maps, dim=0)
        tq_maps = [F.interpolate(
            tq_maps, scale_factor=0.5**(i+1), mode='nearest')for i in range(1)]
        # tq_maps = [tq_maps]
    else:
        tq_maps = [torch.Tensor([0])]

    metrics["voxel_gene_time"] = time.time() - t1
    example = {
        'lidar_seqs': points_list,
        'hier_points': hier_points,
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": num_voxels,
        "metrics": metrics,
        "frame_inds": input_dict["frame_idx"],
        "seq_inds": input_dict["seq_idx"],
        "pose_seq": input_dict["pose_seq"],
        "odometry": input_dict["odometry"],
        "skip": input_dict["skip"],
        "tq_maps": tq_maps,
        "icp_odometry": input_dict.get("icp_odometry", np.zeros(7) ),
        # "normal_gt_seq":normal_gt_list,
    }

    metrics["prep_time"] = time.time() - t

    # for RT in input_dict["cam_pose_seq"]:
    #     example["cam_pose_seq"].append(RT_to_qt(RT))

    # example["odometry"] = compute_rel_pose(
    #     example["cam_pose_seq"][-2], example["cam_pose_seq"][-1])

    return example
