import pathlib
import pickle
import time
from functools import partial

import numpy as np
import torch
import rslo.utils.pose_utils as tch_p

# from rslo.core import box_np_ops
# from rslo.core import preprocess as prep
from rslo.data import kitti_common as kitti

REGISTERED_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def get_dataset_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


class Dataset(object):
    NumPointFeatures = -1

    def __getitem__(self, index):
      
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_sensor_data(self, query):
      
        raise NotImplementedError

    def evaluation(self, dt_annos, output_dir):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError



def generate_pointwise_local_transformation_tch(tq, spatial_size, origin_loc, voxel_size, inv_trans_factor=-1, ):
    '''
    x up, y left 
    x right, y up
    '''
    # assert(len(spatial_size) == 2)
    # if isinstance(tq, np.ndarray):
    device= tq.device
    dtype = tq.dtype
    # device=torch.device(type='cuda', index=0)
    # tq = torch.from_numpy(tq.astype(np.float32))
    # tq=tq.to(device=device)
    t_g = tq[:3]
    q_g = tq[3:]

    if len(spatial_size) == 2:
        size_x, size_y = spatial_size
        size_z = size_x//size_x 
    elif len(spatial_size) == 3:
        size_x, size_y, size_z = spatial_size
        # size_z = 1
    else:
        raise ValueError()

    # generate coordinates grid

    iv, jv, kv = torch.meshgrid(torch.arange(size_y,device=device), torch.arange(
        size_x,device=device), torch.arange(size_z,device=device))  # (size_x,size_y)
    # move the origin to the middle
    # minus 0.5 to shift to the centre of each grid
    # xv = (jv - origin_loc[0]+0.5)*voxel_size[0]
    # yv = (-iv + origin_loc[1]-0.5)*voxel_size[1]
    xv = (jv - origin_loc[0])*voxel_size[0]
    yv = (-iv + origin_loc[1])*voxel_size[1]
    # zv = np.zeros_like(xv)
    zv = (kv-origin_loc[2])*voxel_size[2]

    # xyzv = np.stack([xv, yv, zv], axis=0).reshape([-1, 3])  # Nx3
    # Nx3 # fixed on 7/11/2019
    xyzv = torch.stack([xv, yv, zv], dim=-1).reshape([-1, 3]).to(dtype=dtype)

    if inv_trans_factor > 0:
        xyzv[:, :2] = inv_trans_factor / \
            (np.linalg.norm(xyzv[:, :2], axis=1,
                            keepdims=True)+0.1) ** 2 * xyzv[:, :2]

    t_l = tch_p.rotate_vec_by_q(t=t_g[None, ...]-xyzv, q=tch_p.qinv(
        q_g[None, ...]).repeat([xyzv.shape[0],1]) ) + xyzv

    # t_map = t_l.reshape([size_x, size_y, 3])
    # t_map = t_l.reshape([size_y, size_x, 3])
    #bug fixed on 8/1/2020
    t_map = t_l.reshape([size_y, size_x, size_z, 3])
    # t_map = t_l.reshape([size_x, size_y, size_z, 3])

    # q_map = np.ones([size_y, size_x, 4], np.float32)*q_g
    #bug fixed on 8/1/2020
    q_map = torch.ones([size_y, size_x, size_z, 4], dtype=dtype, device=device)*q_g
    # q_map = np.ones([size_x, size_y, size_z, 4], np.float32)*q_g

    # tq_map = np.concatenate([t_map, q_map], axis=-1).transpose([2, 0, 1])
    tq_map = torch.cat(
        [t_map, q_map], dim=-1).permute(3, 2, 0, 1).squeeze()  # channel, z,y,x
    # print(tq_map.shape,'!!!')
    return tq_map#torch.from_numpy(tq_map)




def from_pointwise_local_transformation_tch(tq_map,  pc_range, inv_trans_factor=-1):
    '''
    x up, y left 
    '''

    dtype = tq_map.dtype
    device = tq_map.device
    input_shape = tq_map.shape
    spatial_size = input_shape[2:]
    if len(spatial_size)==2:
        spatial_size = [1]+list(spatial_size)

    grid_size = torch.from_numpy(
        np.array(list(spatial_size[::-1]))).to(device=device, dtype=dtype)  # attention
    # voxel_size = np.array(voxel_generator.voxel_size)*4  # x,y,z
    pc_range = torch.from_numpy(pc_range).to(device=device, dtype=dtype)

    # voxel_size = (pc_range[3:5]-pc_range[0:2])/grid_size
    voxel_size = (pc_range[3:]-pc_range[:3])/grid_size

    # origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*grid_size[0], (pc_range[4]-(
    #     pc_range[4]+pc_range[1])/2)/(pc_range[4]-pc_range[1])*grid_size[1], 0
    # #fixed on 7/11/2019 the left-top of a grid is the anchor point

    origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*(grid_size[0]
                                                            ), (pc_range[4]-0)/(pc_range[4]-pc_range[1])*(grid_size[1]),  (0-pc_range[2])/(pc_range[5]-pc_range[2]) * grid_size[2]

    assert len(input_shape) == 4
    # tq_map = torch.cat([t_map, q_map], dim=-1).permute([2, 0, 1])
    # tq_map = tq_map.permute([1,2, 0])
    #bug fixed on 8/1/2020
    tq_map = tq_map.permute([0, 2, 3, 1]).contiguous() #b,y,x,c !!!
    # tq_map = tq_map.permute([0, 3, 2, 1]).contiguous() #b,x,y,c
    # t_map = t_l.view([size_x, size_y, 3])
    tq_map = tq_map.view(-1, 7)

    t_l = tq_map[:, :3]
    q_l = tq_map[:, 3:]

    size_z,size_y, size_x= spatial_size
    iv, jv, kv = torch.meshgrid([
        torch.arange(size_y, dtype=dtype, device=device), 
        torch.arange(size_x, dtype=dtype, device=device),  
        torch.arange(size_z, dtype=dtype, device=device )])  # (size_x,size_y)
    # move the origin to the middle
    # minus 0.5 to shift to the centre of each grid
    # xv = (jv - origin_loc[0]+0.5)*voxel_size[0]
    # yv = (-iv + origin_loc[1]-0.5)*voxel_size[1]
    xv = (jv - origin_loc[0])*voxel_size[0]
    yv = (-iv + origin_loc[1])*voxel_size[1]
    zv = (kv-origin_loc[2])*voxel_size[2]
    # zv = torch.zeros_like(xv)

    # xyzv = torch.stack([xv, yv, zv], dim=0).reshape([-1, 3])  # Nx3
    # Nx3 # fixed on 7/11/2019
    xyzv = torch.stack([xv, yv, zv], dim=-1).reshape([-1, 3])
    if inv_trans_factor > 0:
        xyzv[:, :2] = inv_trans_factor / \
            (torch.norm(xyzv[:, :2], dim=1, keepdim=True) +
             0.1) ** 2 * xyzv[:, :2]

    xyzv = torch.cat([xyzv]*input_shape[0], dim=0)

    # import pdb
    # pdb.set_trace()

    t_g = tch_p.rotate_vec_by_q(t=(t_l-xyzv), q=q_l) + xyzv

    #bug fixed on 8/1/2020
    t_map_g = t_g.view(input_shape[0], input_shape[2], input_shape[3], 3) #b,y,x,c
    # t_map_g = t_g.view(input_shape[0], input_shape[3], input_shape[2], 3) #b,x,y,c

    # torch.ones([spatial_size[0], spatial_size[1], 4], np.float32)*q_g
    #bug fixed on 8/1/2020
    q_map_g = q_l.view(input_shape[0], input_shape[2], input_shape[3], 4)
    # q_map_g = q_l.view(input_shape[0], input_shape[3], input_shape[2], 4)

    q_map_g = torch.nn.functional.normalize(q_map_g, dim=-1)

    # tq_map_g = torch.cat([t_map_g, q_map_g], dim=-
    #                      1).permute([0, 3, 1, 2]).contiguous()
    #bug fixed on 8/1/2020
    tq_map_g = torch.cat([t_map_g, q_map_g], dim=-
                         1).permute([0, 3, 1, 2]).contiguous()
    # tq_map_g = torch.cat([t_map_g, q_map_g], dim=-
    #                      1).permute([0, 3, 2, 1]).contiguous() #b,c,y,x

    return tq_map_g