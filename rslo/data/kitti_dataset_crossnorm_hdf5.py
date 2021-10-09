import os
import h5py
from pathlib import Path
import pickle
import time
from functools import partial

import numpy as np
import random

# from rslo.core import box_np_ops
# from rslo.core import preprocess as prep
from rslo.data import kitti_common as kitti
# from rslo.utils.eval import get_coco_eval_result, get_official_eval_result
from rslo.data.dataset import Dataset, register_dataset
from rslo.utils.progress_bar import progress_bar_iter as prog_bar
# , compute_rel_pose
from rslo.utils.geometric import RT_to_tq, cam_pose_to_lidar, expand_rigid_transformation, odom_to_abs_pose
from rslo.utils.visualization import draw_trajectory, pltfig2data
from rslo.utils.pose_utils_np import *
import rslo.utils.pose_utils as tch_p
from rslo.utils.kitti_evaluation import kittiOdomEval

from collections import defaultdict
from utils.singleton import HDF5File

@register_dataset
class KittiDatasetCrossNormalHDF5(Dataset):
    # NumPointFeatures = 4

    def __init__(self,
                 root_path,
                 info_path,
                 seq_length=2,
                 skip=1,
                 random_skip=False,
                 split='train',
                 prep_func=None,
                 cycle_constraint=False,
                 num_point_features=None,
                 step=1,
                 ):
        assert info_path is not None
        assert split in ['train', 'val', 'eval_train','test', 'train+test', ]
        # with open(info_path, 'rb') as f:
        #     infos = pickle.load(f)
        self._root_path = Path(root_path)

        self.odom_eval = kittiOdomEval()
        self._prep_func = prep_func
        self._local_seq_length = seq_length
        # assert skip >= 1
        assert skip != 0
        self._skip = skip
        self._random_skip = random_skip
        self._actual_skip = skip
        self.seq_segments = None
        assert step > 0
        self._step = step
        self.NumPointFeatures = num_point_features

        self.dataset=None
        self.file_path = info_path
        if split == 'train':
            self.seqs = [f'{i:02d}' for i in range(7)]
        elif split == 'val':
            self.seqs = [f'{i:02d}' for i in range(7, 11)]
        elif split == 'eval_train':
            self.seqs = [f'{i:02d}' for i in range(0,1)]
        elif split == 'train+test':
            self.seqs = [f'{i:02d}' for i in range(7)] + [f'{i:02d}' for i in range(11,22)]

        print("seqs:",self.seqs) 

        # self._kitti_infos = kitti_info_convertor(infos)
        self.info_preprocess(self.file_path, self.seqs)
        print("Dataset", self.seqs, self.seq_lens, flush=True)
        #TODO: 
        with open(os.path.join(root_path, 'icp_tmp_all.pkl'), 'rb') as f:
            self.icp_pose  = pickle.load(f)

    def info_preprocess(self, h5_file_path, seqs):
        self.seq_lens = []
        # with h5py.File(h5_file_path, 'r', libver='latest',swmr=True) as f:
        #     assert f.swmr_mode
        f= HDF5File(h5_file_path, 'r', libver='latest', swmr=True).read()

        for seq in seqs:
            seq_len = len(f[seq]['lidar_points'])
            self.seq_lens.append(seq_len)
    def __len__(self):
        # return (np.sum(self._kitti_infos['seq_lengths'])+self._step-1)//self._step
        return (np.sum(self.seq_lens)+self._step-1)//self._step

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list)):
            idx, seed = idx
            # import pdb
            # pdb.set_trace()
        else:
            seed = None

        # if self.dataset is None:
        #     # self.dataset = h5py.File(self.file_path, 'r', libver='latest', swmr=True)
        #     self.dataset= HDF5File(self.file_path, 'r', libver='latest', swmr=True).read()
        #     assert self.dataset.swmr_mode


        input_dict = self.get_sensor_data(idx)

        example = self._prep_func(input_dict=input_dict, seed=seed)
        example['velodyne_path'] = input_dict['velodyne_path']
        # example["metadata"] = {}
        # if "image_idx" in input_dict["metadata"]:
        #     example["metadata"] = input_dict["metadata"]
        # if "anchors_mask" in example:
        #     example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def _get_info_from_idx(self, idx):
        idx = self._step*idx
        # seq_lengths = np.array(self._kitti_infos['seq_lengths'])
        seq_lengths = np.array(self.seq_lens)
        seq_lengths_cum = np.cumsum(seq_lengths)
        seq_lengths_cum = np.insert(seq_lengths_cum, 0, 0)  # insert a dummy 0
        seq_idx = np.nonzero(seq_lengths_cum > idx)[0][0]-1

        frame_idx = idx - seq_lengths_cum[seq_idx]
        # info = self._kitti_infos["data"][seq_idx][frame_idx]
        # abs_seq_idx = seq_idx + int(self.seqs[0])
        abs_seq_idx = int(self.seqs[seq_idx])

        return abs_seq_idx 

    def get_sensor_data(self, query, with_lidar=True):
        """
            kitti_infos: {
                "seq_lengths":[list], the lengths for each sequence.
                "data":[  # the data information for each sequence
                    [
                        # seq 0
                        [
                            # 1st frame in seq 0
                            {
                            "seq": [int], sequence_idx,
                            "frame_idx": [int], frame index,
                            "lidar_path": [str],
                            "img_path": [str],
                            "cam_pose": [np.array], 3x4
                            "calib/P1": [np.array], 3x4
                            "calib/P2": [np.array], 3x4
                            "calib/P3": [np.array], 3x4
                            "calib/P4": [np.array], 3x4
                            "calib/Tr_velo_to_cam": [np.array], 4x4
                        },
                        ...
                        ]
                    ],
                    ...
                ]

            }
        """



        query = self._step*query

        idx = query
        # seq_lengths = np.array(self._kitti_infos['seq_lengths'])
        seq_lengths = np.array(self.seq_lens)
        seq_lengths_cum = np.cumsum(seq_lengths)
        seq_lengths_cum = np.insert(seq_lengths_cum, 0, 0)  # insert a dummy 0
        # seq_idx = np.nonzero(seq_lengths_cum > idx)[0][0]-1 
        seq_idx = np.nonzero(seq_lengths_cum > idx)[0][0]-1  
        # abs_seq_idx = seq_idx + int(self.seqs[0]) # add a base seq idx
        abs_seq_idx = int(self.seqs[seq_idx])

        frame_idx = idx - seq_lengths_cum[seq_idx]


        # info = self._kitti_infos["data"][seq_idx]
        # if self.dataset is None:
            # self.dataset = h5py.File(self.file_path, 'r', libver='latest', swmr=True)
        dataset= HDF5File(self.file_path, 'r', libver='latest', swmr=True, rdcc_nbytes=1024**2*10).read()
        assert dataset.swmr_mode
        info = dataset[f"{abs_seq_idx:02d}"]

        if self._random_skip:
            # skip = random.randint(1 , self._skip )
            if self._skip > 0:
                choices = list(np.arange(1, self._skip+1))
            else:
                choices = list(np.arange(self._skip, 0)) + \
                    list(np.arange(1, -self._skip+1))
            skip = random.choice(choices)
            # skip = random.randint(min(1,self._skip) , max(0, self._skip) )
        else:
            skip = self._skip

        local_seq = np.array([frame_idx -
                              i for i in range(skip*(self._local_seq_length-1), -np.sign(skip), -skip)], dtype=int)
        # set the negative index to 0
        local_seq = np.where(local_seq < 0, 0, local_seq)
        # local_seq = np.where(local_seq > len(
        #     info)-1, len(info)-1, local_seq)  # added at 14/12/09
        local_seq = np.where(local_seq > len(
            info['lidar_points'])-1, len(info['lidar_points'])-1, local_seq)  # added at 14/12/09

        res = {
            "skip": [skip]*self._local_seq_length,
            "seq_idx": [],
            "frame_idx": [],
            "lidar_seq": [],
            "pose_seq": [],
            "odometry": None,
            "calib/Tr_velo_to_cam": [],
            "velodyne_path": [],
            "hier_points_seq":[],
            "icp_pose_seq":[]
        }

        for i in local_seq:
            if i==4148 and abs_seq_idx==19:
                #TODO
                #skip current frame
                i=i+1
            # velo_path = Path(info[i]['velodyne_path'])
            # if self.NumPointFeatures == 4:
            #     velo_path = velo_path
            # elif self.NumPointFeatures == 8:
            #     velo_path = velo_path.parent.parent / "bin_normal"/velo_path.name
            # elif self.NumPointFeatures == 9:
            #     velo_path = velo_path.parent.parent / "chamfer"/velo_path.name
            # elif self.NumPointFeatures == 7:
            #     velo_path = velo_path.parent.parent / "normal_spherical"/velo_path.name
            # else:
            #     raise ValueError()

            # if not velo_path.is_absolute():
            #     # velo_path = Path(self._root_path) / info[i]['velodyne_path']
            #     velo_path = Path(self._root_path) / \
            #         velo_path  # info[i]['velodyne_path']

            if with_lidar:
                # points = np.fromfile(
                #     str(velo_path), dtype=np.float32,
                #     count=-1).reshape([-1, self.NumPointFeatures])  # read points from file
                if self.NumPointFeatures==6:
                    # print(info['lidar_points'][i].reshape([-1,4]).max(axis=0), '!!!' )
                    lidar_points = info['lidar_points'][i].reshape([-1,4])[:,:3]
                else:
                    lidar_points = info['lidar_points'][i].reshape([-1,4])
                    
                    # lidar_points = info['lidar_points'][i].reshape([-1,4])

                # lidar_normals = info['lidar_normals'][i].reshape([-1,3])
                lidar_normals = info['lidar_cross_normals'][i].reshape([-1,3])
                
                points = np.concatenate([lidar_points, lidar_normals], axis=-1)
                #
                # points[:, -3:][np.abs( points[:, -3:])==[0,0,1] ] = 0 

                # normal_gt=np.fromfile(
                #     str(velo_path), dtype=np.float32,
                #     count=-1).reshape([-1, self.NumPointFeatures])[:,4:]
                # '''
                normal_gt=info['lidar_normals'][i].reshape([-1, 3])
                normal_gt[:, -3:][np.abs( normal_gt[:, -3:])==[0,0,1] ] = 0 

                res["lidar_seq"].append(np.concatenate([points, normal_gt], axis=-1 ) )
                # '''
                # res["lidar_seq"].append(points   )

                # hier_points = [info[f'hier_lidar_points_normals_{vs}'][i].reshape([-1,6]) for vs in [0.1,0.2,0.4,0.8]]#
                # hier_points = [info[f'hier_lidar_points_normals_{vs}'][i].reshape([-1,6]) for vs in [0.1]]
                #TODO: dummy value
                res["hier_points_seq"].append([np.zeros_like(points[:1])] ) 
            else:
                points = None
                hier_points=None
                res["lidar_seq"].append(points)

                res["hier_points_seq"].append(hier_points)

            try:
                res["pose_seq"].append(  # transform to lidar system
                    RT_to_tq(
                        cam_pose_to_lidar(
                            # info[i]["cam_pose"], info[i]["calib/Tr_velo_to_cam"]
                            info["poses"][i], info["calib.Tr_velo_to_cam"][i]
                        )
                    ))
            except:
                # if abs_seq_idx==19 and frame_idx==4148:
                    
                #     res["pose_seq"].append(  # transform to lidar system
                #         RT_to_tq(
                #             cam_pose_to_lidar(
                #                 info["poses"][i+1], info["calib.Tr_velo_to_cam"][i+1]
                #             )
                #         )
                #     )
                    
                print(info["poses"][i], info["calib.Tr_velo_to_cam"][i],'!!',abs_seq_idx,frame_idx,i )
            # res["seq_idx"].append(np.array([info[i]["seq"]]))
            # res["icp_pose_seq"].append(self.icp_pose[abs_seq_idx][i:i+1])
            #TODO:
            # res["icp_pose_seq"].append(np.array([0,0,0,1,0,0,0]).reshape([-1,7]) )
            res["seq_idx"].append(np.array([abs_seq_idx]))
            res["frame_idx"].append(np.array([frame_idx]))
            res["calib/Tr_velo_to_cam"].append(info["calib.Tr_velo_to_cam"][i])
            # res['velodyne_path'].append(info[i]['velodyne_path']) 


        # res["odometry"] = compute_rel_pose(
        #     res["cam_pose_seq"][-2], res["cam_pose_seq"][-1])
        # res["odometry"] = calc_vo(
        #     res["pose_seq"][-2], res["pose_seq"][-1]).squeeze(axis=0)
        res["odometry"] = self.generate_cyc_vo(res["pose_seq"])
        # res["icp_odometry"] = self.generate_cyc_vo(res["icp_pose_seq"])

        return res

    def generate_cyc_vo(self, pose_seq):
        assert len(pose_seq) > 1
        seq_len = len(pose_seq)

        vos = []
        for i in range(0, seq_len):
            for j in range(i+1, seq_len):
                # x1.append(xs[i])
                # x2.append(xs[j])
                vo = calc_vo(pose_seq[i], pose_seq[j]).squeeze(axis=0)
                vo[3:] *= np.sign(vo[3])
                # vos.append(calc_vo(pose_seq[i], pose_seq[j]).squeeze(axis=0))
                vos.append(vo)

        vos = np.stack(vos, axis=0)  # (7*(seq_len(seq_len-1)/2),)
        return vos

    def evaluation(self, prediction, output_dir):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        If you want to eval by my KITTI eval function, you must
        provide the correct format prediction.
         prediction =[ {
            "translation_preds": x[:, :3],
            "rotation_preds": x[:, 3:],
        },
        ...
        ]

        """
        # TODO: transform the odometry to camera coordinates
        gts = np.stack([self.get_sensor_data(i)['odometry'].reshape(-1)
                        for i in range(self.__len__())], axis=0)
        # gts = np.concatenate([self.get_sensor_data(i)['pose_seq'][-1]
        #                       for i in range(self.__len__())], axis=0)

        preds = [
            np.concatenate(
                [prediction[i]['translation_preds'].detach().cpu().numpy(), prediction[i]['rotation_preds'].detach().cpu().numpy()], axis=-1) for i in range(len(prediction))
        ]
        preds = np.concatenate(preds, axis=0)

        abs_gts = odom_to_abs_pose(gts)
        abs_preds = odom_to_abs_pose(preds)
        # the same for gts
        # TODO: finish the evaluation
        Errors = self.odom_eval.calcSequenceErrors(abs_preds, abs_gts)
        segmentErrors = self.odom_eval.computeSegmentErr(Errors)
        avgSegmentErrors = self.odom_eval.computeSegmentAvgErr(segmentErrors)

        # generate odometry plot
        # fig, ax = draw_odometry(preds)
        # fig, ax = draw_odometry(gts, figure=fig, ax=ax, color='r')
        fig, ax = draw_trajectory(abs_preds, abs_gts, color='b')
        odom_plot = pltfig2data(fig)

        return {
            "error": {
                "kitti_error": segmentErrors,
                "kitti_avg_error": avgSegmentErrors,
            },
            "plot": {
                "odometry_pred": odom_plot[:, :, :3].transpose([2, 0, 1])[np.newaxis, :]
            }
        }

    def evaluation_seqs(self, prediction, output_dir):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        If you want to eval by my KITTI eval function, you must
        provide the correct format prediction.
        prediction =[ {
            "translation_preds": x[:, :3],
            "rotation_preds": x[:, 3:],
        },
        ...
        ]

        """
        # obtain the first and last index of all sequences
        # assume the same sequence is continuous
        if self.seq_segments is None:
            self.seq_segments = defaultdict(list)
            for i in range(self.__len__()):
                # seq_idx = int(self.get_sensor_data(i)['seq_idx'][0])
                # seq_idx = int(self._get_info_from_idx(i)['seq'])
                seq_idx = int(self._get_info_from_idx(i))
                if seq_idx not in self.seq_segments:
                    self.seq_segments[seq_idx] = [i, i]
                else:
                    self.seq_segments[seq_idx][0],  self.seq_segments[seq_idx][1] = min(
                        self.seq_segments[seq_idx][0], i),  max(self.seq_segments[seq_idx][1], i)
        # print(self.seq_segments, '!!!')
        # TODO: transform the odometry to camera coordinates
        gts = np.stack([self.get_sensor_data(i, with_lidar=False)['odometry'].reshape(-1)
                        for i in range(self.__len__())], axis=0)
        if prediction.shape[-1]==6: #logq
           prediction=tch_p.qexp_t(prediction)
        
        preds=prediction.detach().cpu().numpy()

        if len(preds) < len(gts):
            print("Warning: the length of prediction is smaller than that of GT!!!")
            gts = gts[:len(preds)]

        abs_gts = {}
        abs_preds = {}
        for seq_idx, seg in self.seq_segments.items():
            if seg[0]>=len(gts):
                print("Warning: the length of prediction is smaller than that 800!!!")
                continue 
            abs_gts[seq_idx] = odom_to_abs_pose(gts[seg[0]:seg[1]+1 ])

            abs_preds[seq_idx] = odom_to_abs_pose(preds[seg[0]:seg[1]+1 ])
        # the same for gts
        # TODO: finish the evaluation
        # Errors = self.odom_eval.calcSequenceErrors(abs_preds, abs_gts)
        # segmentErrors = self.odom_eval.computeSegmentErr(Errors)
        # avgSegmentErrors = self.odom_eval.computeSegmentAvgErr(segmentErrors)

        seq_errors = {
            "translation_error": defaultdict(dict),
            "rotation_error": defaultdict(dict),

        }
        odom_errs = {}
        for seq_idx, seg in self.seq_segments.items():
            if abs_preds.get(seq_idx) is None:
                continue
            Errors = self.odom_eval.calcSequenceErrors(
                abs_preds[seq_idx], abs_gts[seq_idx])
            segmentErrors,seg_err_seq = self.odom_eval.computeSegmentErr(Errors, return_seg_err=True)
            avgSegmentErrors = self.odom_eval.computeSegmentAvgErr(
                segmentErrors)

            odom_errs[seq_idx] = self.odom_eval.calcOdomErrors(preds, gts)

            seq_errors["translation_error"][seq_idx]['seg_error'] = dict(zip(
                segmentErrors.keys(), [segmentErrors[k][0] for k in segmentErrors.keys()]))

            seq_errors["rotation_error"][seq_idx]['seg_error'] = dict(zip(
                segmentErrors.keys(), [segmentErrors[k][1] for k in segmentErrors.keys()]))

            seq_errors["translation_error"][seq_idx]['avg_seg_error'] = avgSegmentErrors[0]
            seq_errors["rotation_error"][seq_idx]['avg_seg_error'] = avgSegmentErrors[1]

        seq_errors["avg_translation_error"] = np.mean([seq_errors["translation_error"][i]["avg_seg_error"]
                                                       for i in seq_errors["translation_error"].keys()])
        seq_errors["avg_rotaion_error"] = np.mean([seq_errors["rotation_error"][i]["avg_seg_error"]
                                                   for i in seq_errors["rotation_error"].keys()])
        # generate odometry plot
        # fig, ax = draw_odometry(preds)
        # fig, ax = draw_odometry(gts, figure=fig, ax=ax, color='r')
  
        plots = {}
        for seq_idx in abs_preds:

            # fig, ax = draw_trajectory(
            #     abs_preds[seq_idx], abs_gts[seq_idx], color='b', error_step=10, errors=np.array(seg_err_seq[100]))
            fig, ax = draw_trajectory(
                abs_preds[seq_idx], abs_gts[seq_idx], color='b', error_step=20, odom_errors=np.array( odom_errs[seq_idx]) )
            odom_plot = pltfig2data(fig)
            plots[seq_idx] = odom_plot[:, :, :3].transpose([2, 0, 1])[
                np.newaxis, :]

        #TODO: tmp solution
        with open("./icp_tmp_val.pkl", 'wb') as f:
            print(len(abs_preds))
            pickle.dump(abs_preds, f)
        with open("./val_gt.pkl", 'wb') as f:
            pickle.dump(abs_gts, f)
        return {
            "error": {
                "kitti_error": seq_errors,
                # "kitti_avg_error": avgSegmentErrors,
            },
            "plot": {
                # odom_plot[:, :, :3].transpose([2, 0, 1])[np.newaxis, :]
                "odometry_pred": plots
            }
        }

def generate_pointwise_local_transformation_tch(tq, spatial_size, origin_loc, voxel_size, inv_trans_factor=-1, ):
    '''
    x up, y left ? 
    x right, y up
    '''
    # assert(len(spatial_size) == 2)
    # if isinstance(tq, np.ndarray):
    device= tq.device
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
    xyzv = torch.stack([xv, yv, zv], dim=-1).reshape([-1, 3])

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
    q_map = torch.ones([size_y, size_x, size_z, 4], dtype=torch.float32, device=device)*q_g
    # q_map = np.ones([size_x, size_y, size_z, 4], np.float32)*q_g

    # tq_map = np.concatenate([t_map, q_map], axis=-1).transpose([2, 0, 1])
    tq_map = torch.cat(
        [t_map, q_map], dim=-1).permute(3, 2, 0, 1).squeeze()  # channel, z,y,x
    # print(tq_map.shape,'!!!')
    return tq_map#torch.from_numpy(tq_map)
    
def generate_pointwise_local_transformation(tq, spatial_size, origin_loc, voxel_size, inv_trans_factor=-1, ):
    '''
    x up, y left ? 
    x right, y up
    '''
    # assert(len(spatial_size) == 2)
    t_g = tq[:3]
    q_g = tq[3:]

    if len(spatial_size) == 2:
        size_x, size_y = spatial_size
        size_z = 1
    elif len(spatial_size) == 3:
        size_x, size_y, size_z = spatial_size
        # size_z = 1
    else:
        raise ValueError()

    # generate coordinates grid
    iv, jv, kv = np.meshgrid(range(size_y), range(
        size_x), range(size_z), indexing='ij')  # (size_x,size_y)
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
    xyzv = np.stack([xv, yv, zv], axis=-1).reshape([-1, 3])

    if inv_trans_factor > 0:
        xyzv[:, :2] = inv_trans_factor / \
            (np.linalg.norm(xyzv[:, :2], axis=1,
                            keepdims=True)+0.1) ** 2 * xyzv[:, :2]

    t_l = rotate_vec_by_q(t=t_g[np.newaxis, ...]-xyzv, q=qinv(
        q_g[np.newaxis, ...])) + xyzv

    # t_map = t_l.reshape([size_x, size_y, 3])
    # t_map = t_l.reshape([size_y, size_x, 3])
    #bug fixed on 8/1/2020
    t_map = t_l.reshape([size_y, size_x, size_z, 3])
    # t_map = t_l.reshape([size_x, size_y, size_z, 3])

    # q_map = np.ones([size_y, size_x, 4], np.float32)*q_g
    #bug fixed on 8/1/2020
    q_map = np.ones([size_y, size_x, size_z, 4], np.float32)*q_g
    # q_map = np.ones([size_x, size_y, size_z, 4], np.float32)*q_g

    # tq_map = np.concatenate([t_map, q_map], axis=-1).transpose([2, 0, 1])
    tq_map = np.concatenate(
        [t_map, q_map], axis=-1).transpose([3, 2, 0, 1]).squeeze()  # channel, z,y,x
    # print(tq_map.shape,'!!!')
    return torch.from_numpy(tq_map)


def generate_pointwise_local_transformation_3d(tq, spatial_size, origin_loc, voxel_size, inv_trans_factor=-1, ):
    '''
    x up, y left ? 
    x right, y up
    '''
    # assert(len(spatial_size) == 2)
    t_g = tq[:3]
    q_g = tq[3:]

    if len(spatial_size) == 2:
        size_x, size_y = spatial_size
        size_z = 1
    elif len(spatial_size) == 3:
        size_x, size_y, size_z = spatial_size
        # size_z = 1
    else:
        raise ValueError()

    # generate coordinates grid
    iv, jv, kv = np.meshgrid(range(size_y), range(
        size_x), range(size_z), indexing='ij')  # (size_x,size_y)
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
    xyzv = np.stack([xv, yv, zv], axis=-1).reshape([-1, 3])

    if inv_trans_factor > 0:
        xyzv[:, :2] = inv_trans_factor / \
            (np.linalg.norm(xyzv[:, :2], axis=1,
                            keepdims=True)+0.1) ** 2 * xyzv[:, :2]

    t_l = rotate_vec_by_q(t=t_g[np.newaxis, ...]-xyzv, q=qinv(
        q_g[np.newaxis, ...])) + xyzv

    # t_map = t_l.reshape([size_x, size_y, 3])
    # t_map = t_l.reshape([size_y, size_x, 3])
    #bug fixed on 8/1/2020
    t_map = t_l.reshape([size_y, size_x, size_z, 3])
    # t_map = t_l.reshape([size_x, size_y, size_z, 3])

    # q_map = np.ones([size_y, size_x, 4], np.float32)*q_g
    #bug fixed on 8/1/2020
    q_map = np.ones([size_y, size_x, size_z, 4], np.float32)*q_g
    # q_map = np.ones([size_x, size_y, size_z, 4], np.float32)*q_g

    # tq_map = np.concatenate([t_map, q_map], axis=-1).transpose([2, 0, 1])
    tq_map = np.concatenate(
        [t_map, q_map], axis=-1).transpose([3, 2, 0, 1]).squeeze()  # channel, z,y,x
    # print(tq_map.shape, '!!!!!')
    
    return torch.from_numpy(tq_map)




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


def from_pointwise_local_transformation_tch_3d(tq_map,  pc_range, inv_trans_factor=-1):
    '''
    x up, y left 
    '''


    if len(tq_map.shape)==4:
        tq_map = tq_map[:,:,None]
    dtype = tq_map.dtype
    device = tq_map.device
    input_shape = tq_map.shape
    spatial_size = input_shape[2:] #b,c,z,y,x

    grid_size = torch.from_numpy(
        np.array(list(spatial_size[::-1]))).to(device=device, dtype=dtype)  # attention!
    # voxel_size = np.array(voxel_generator.voxel_size)*4  # x,y,z
    pc_range = torch.from_numpy(pc_range).to(device=device, dtype=dtype)

    voxel_size = (pc_range[3:]-pc_range[:3])/grid_size  # x,y,z

    # origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*grid_size[0], (pc_range[4]-(
    #     pc_range[4]+pc_range[1])/2)/(pc_range[4]-pc_range[1])*grid_size[1], 0
    # #fixed on 7/11/2019 the left-top of a grid is the anchor point
    origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*(grid_size[0]
                                                            ), (pc_range[4]-0)/(pc_range[4]-pc_range[1])*(grid_size[1]), (0-pc_range[2])/(pc_range[5]-pc_range[2]) * grid_size[2]

    assert len(input_shape) == 5

    # tq_map = tq_map.permute([0, 2, 3, 1]).contiguous()
    # import  pdb 
    # pdb.set_trace()
    tq_map = tq_map.permute([0, 3, 4, 2, 1]).contiguous() #b,c,z,y,x->b,y,x,z,c
    # t_map = t_l.view([size_x, size_y, 3])
    tq_map = tq_map.view(-1, 7)

    t_l = tq_map[:, :3]
    q_l = tq_map[:, 3:]

    # size_y, size_x = spatial_size
    size_z,size_y, size_x = spatial_size

    iv, jv, kv= torch.meshgrid([
        torch.arange(size_y, dtype=dtype, device=device), 
        torch.arange(size_x, dtype=dtype, device=device), 
        torch.arange(size_z, dtype=dtype, device=device), 
        ])  # (size_x,size_y)
    # move the origin to the middle
    # minus 0.5 to shift to the centre of each grid
    # xv = (jv - origin_loc[0]+0.5)*voxel_size[0]
    # yv = (-iv + origin_loc[1]-0.5)*voxel_size[1]
    xv = (jv - origin_loc[0])*voxel_size[0]
    yv = (-iv + origin_loc[1])*voxel_size[1]
    # zv = torch.zeros_like(xv)
    zv = (kv-origin_loc[2])*voxel_size[2]

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

    # t_map_g = t_g.view(input_shape[0], input_shape[2], input_shape[3], 3)
    t_map_g = t_g.view(input_shape[0], input_shape[3], input_shape[4], input_shape[2], 3) #b,y,x,z,c

    q_map_g = q_l.view(input_shape[0], input_shape[3], input_shape[4], input_shape[2] ,4)
    q_map_g = torch.nn.functional.normalize(q_map_g, dim=-1)

    tq_map_g = torch.cat([t_map_g, q_map_g], dim=-
                         1).permute([0, 4, 3, 1,2]).contiguous() #b,y,x,z,c->b,c, z,y,x

    return tq_map_g

def kitti_info_convertor(info, seqs=range(0, 9)):
    """
    [Transform the original kitti info file]
    """

    seqs = info.keys()
    seq_lengths = [len(info[i]) for i in seqs]
    data = []
    for seq in seqs:
        frames = []
        for frame in info[seq]:
            frames.append(
                {
                    "seq":  seq,
                    "frame_idx":  frame['index'],
                    "velodyne_path": frame['lidar_bin_path'],
                    "img_path": None,
                    "cam_pose": frame['pose'],
                    "calib/P0": frame['calib/P0'],
                    "calib/P1": frame['calib/P1'],
                    "calib/P2": frame['calib/P2'],
                    "calib/P3": frame['calib/P3'],
                    "calib/Tr_velo_to_cam": frame['calib/Tr_velo_to_cam']
                })

        data.append(frames)

    kitti_infos = {
        "seq_lengths": seq_lengths,
        "data": data
    }
    return kitti_infos


def _cam_pose_to_lidar(cam_pose, T_velo_to_cam):
    pass

    # def convert_to_kitti_info_version2(info):
    #     """convert kitti info v1 to v2 if possible.
    #     """
    #     if "image" not in info or "calib" not in info or "point_cloud" not in info:
    #         info["image"] = {
    #             'image_shape': info["img_shape"],
    #             'image_idx': info['image_idx'],
    #             'image_path': info['img_path'],
    #         }
    #         info["calib"] = {
    #             "R0_rect": info['calib/R0_rect'],
    #             "Tr_velo_to_cam": info['calib/Tr_velo_to_cam'],
    #             "P2": info['calib/P2'],
    #         }
    #         info["point_cloud"] = {
    #             "velodyne_path": info['velodyne_path'],
    #         }


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    for info in infos:
        pc_info = info["point_cloud"]
        image_info = info["image"]
        calib = info["calib"]
        if relative_path:
            v_path = str(Path(data_path) / pc_info["velodyne_path"])
        else:
            v_path = pc_info["velodyne_path"]
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info["image_shape"])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)


def create_kitti_info_file(data_path, save_path=None, relative_path=True):
    imageset_folder = Path(__file__).resolve().parent / "ImageSets"
    train_img_ids = _read_imageset_file(str(imageset_folder / "train.txt"))
    val_img_ids = _read_imageset_file(str(imageset_folder / "val.txt"))
    test_img_ids = _read_imageset_file(str(imageset_folder / "test.txt"))

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_train = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / 'kitti_infos_train.pkl'
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    kitti_infos_val = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / 'kitti_infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    filename = save_path / 'kitti_infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

    kitti_infos_test = kitti.get_kitti_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / 'kitti_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)


def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False):
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    for info in prog_bar(kitti_infos):
        pc_info = info["point_cloud"]
        image_info = info["image"]
        calib = info["calib"]

        v_path = pc_info['velodyne_path']
        v_path = Path(data_path) / v_path
        points_v = np.fromfile(
            str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
        rect = calib['R0_rect']
        P2 = calib['P2']
        Trv2c = calib['Tr_velo_to_cam']
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                    image_info["image_shape"])
        if save_path is None:
            save_filename = v_path.parent.parent / (
                v_path.parent.stem + "_reduced") / v_path.name
            # save_filename = str(v_path) + '_reduced'
            if back:
                save_filename += "_back"
        else:
            save_filename = str(Path(save_path) / v_path.name)
            if back:
                save_filename += "_back"
        with open(save_filename, 'w') as f:
            points_v.tofile(f)


def create_reduced_point_cloud(data_path,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    if train_info_path is None:
        train_info_path = Path(data_path) / 'kitti_infos_train.pkl'
    if val_info_path is None:
        val_info_path = Path(data_path) / 'kitti_infos_val.pkl'
    if test_info_path is None:
        test_info_path = Path(data_path) / 'kitti_infos_test.pkl'

    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(
            data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, test_info_path, save_path, back=True)


if __name__ == "__main__":
    fire.Fire()
