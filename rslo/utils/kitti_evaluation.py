
#
# Copyright Qing Li (hello.qingli@gmail.com) 2018. All Rights Reserved.
#
# References: 1. KITTI odometry development kit: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
#             2. A Geiger, P Lenz, R Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012.
#

# from tools.pose_evaluation_utils import quat_pose_to_mat
# import tools.transformations as tr
import matplotlib.backends.backend_pdf
import glob
import argparse
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt

from rslo.utils.geometric import tq_to_RT
# choose other backend that not required GUI (Agg, Cairo, PS, PDF or SVG) when use matplotlib
plt.switch_backend('agg')


class kittiOdomEval(object):
    def __init__(self):

        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)

    def toCameraCoord(self, pose_mat):
        '''
            Convert the pose of lidar coordinate to camera coordinate
        '''
        R_C2L = np.array([[0,   0,   1,  0],
                          [-1,  0,   0,  0],
                          [0,  -1,   0,  0],
                          [0,   0,   0,  1]])
        inv_R_C2L = np.linalg.inv(R_C2L)
        R = np.dot(inv_R_C2L, pose_mat)
        rot = np.dot(R, R_C2L)
        return rot

    def trajectoryDistances(self, poses):
        '''
            Compute the length of the trajectory
            poses dictionary: [frame_idx: pose]
        '''
        dist = [0]
        # sort_frame_idx = sorted(poses.keys())
        sort_frame_idx = list(range(len(poses)))
        for i in range(len(sort_frame_idx)-1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i+1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))
        self.distance = dist[-1]
        return dist

    def rotationError(self, pose_error):
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5*(a+b+c-1.0)
        return np.arccos(max(min(d, 1.0), -1.0))

    def translationError(self, pose_error):
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx**2+dy**2+dz**2)

    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1
    
    def calcOdomErrors(self, odom_pred, odom_gt):
        odom_pred = [tq_to_RT(odom_pred[i], expand=True)
                    for i in range(len(odom_pred))]
        odom_gt = [tq_to_RT(odom_gt[i], expand=True)
                    for i in range(len(odom_gt))]
        errs=[]
        for i  in range(len(odom_gt)-1):
            odom_error = np.dot(np.linalg.inv(
                odom_pred[i]), odom_gt[i] )
        
            r_err = self.rotationError(odom_error)
            t_err = self.translationError(odom_error) 

            errs.append([r_err, t_err]) 
        return errs
    def calcSequenceErrors(self, poses_result, poses_gt):

        poses_gt = [tq_to_RT(poses_gt[i], expand=True)
                    for i in range(len(poses_gt))]
        poses_result = [tq_to_RT(poses_result[i], expand=True)
                        for i in range(len(poses_result))]

        err = []
        self.max_speed = 0
        # pre-compute distances (from ground truth as reference)
        dist = self.trajectoryDistances(poses_gt)
        # every second, kitti data 10Hz
        self.step_size = 10
        # for all start positions do
        # for first_frame in range(9, len(poses_gt), self.step_size):

        for first_frame in range(0, len(poses_gt), self.step_size):
            # for all segment lengths do
            for i in range(self.num_lengths):
                # current length
                len_ = self.lengths[i]
                # compute last frame of the segment
                last_frame = self.lastFrameFromSegmentLength(
                    dist, first_frame, len_)

                # Continue if sequence not long enough
                # if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
                if last_frame == -1 or last_frame >= len(poses_result) or first_frame >= len(poses_result):
                    continue

                # compute rotational and translational errors, relative pose error (RPE)
                pose_delta_gt = np.dot(np.linalg.inv(
                    poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(
                    poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(
                    pose_delta_result), pose_delta_gt)

                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)

                # compute speed
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1*num_frames)   # 10Hz
                if speed > self.max_speed:
                    self.max_speed = speed
                err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
        return err

    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0
        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err

    def computeSegmentErr(self, seq_errs, return_seg_err=False):
        '''
            This function calculates average errors for different segment.
        '''
        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        # Get errors
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])

        # Compute average
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                pass
                # del avg_segment_errs[len_]
                # avg_segment_errs[len_] = [0, 0]
        if not return_seg_err:
            return avg_segment_errs
        else:
            return avg_segment_errs, segment_errs
    def computeSegmentAvgErr(self, segment_errs):
        if len(segment_errs)==0:
            return 0, 0
        avg_t_err = 0
        avg_r_err = 0
        for k, v in segment_errs.items():
            if len(v) < 2:
                print("Error! Key:", k)
            avg_t_err += v[0]
            avg_r_err += v[1]
        return avg_t_err/len(segment_errs), avg_r_err/len(segment_errs)
    def computeSegmentRMSEErr(self, segment_errs):
        if len(segment_errs)==0:
            return 0, 0
        avg_t_err = 0
        avg_r_err = 0
        for k, v in segment_errs.items():
            if len(v) < 2:
                print("Error! Key:", k)
            avg_t_err += v[0]**2
            avg_r_err += v[1]**2

        return np.sqrt(avg_t_err/len(segment_errs)), np.sqrt(avg_r_err/len(segment_errs))

    def computeSpeedErr(self, seq_errs):
        '''
            This function calculates average errors for different speed.
        '''
        segment_errs = {}
        avg_segment_errs = {}
        for s in range(2, 25, 2):
            segment_errs[s] = []

        # Get errors
        for err in seq_errs:
            speed = err[4]
            t_err = err[2]
            r_err = err[1]
            for key in segment_errs.keys():
                if np.abs(speed - key) < 2.0:
                    segment_errs[key].append([t_err, r_err])

        # Compute average
        for key in segment_errs.keys():
            if segment_errs[key] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[key])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[key])[:, 1])
                avg_segment_errs[key] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[key] = []
        return avg_segment_errs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KITTI Evaluation toolkit')
    parser.add_argument('--gt_dir',     type=str, default='./ground_truth_pose',
                        help='Directory path of the ground truth odometry')
    parser.add_argument('--result_dir', type=str, default='./data/',
                        help='Directory path of storing the odometry results')
    parser.add_argument('--eva_seqs',   type=str, default='09_pred,10_pred,11_pred',
                        help='The sequences to be evaluated')
    parser.add_argument('--toCameraCoord',   type=lambda x: (str(x).lower() == 'true'),
                        default=False, help='Whether to convert the pose to camera coordinate')

    args = parser.parse_args()
    pose_eval = kittiOdomEval(args)
    # set the value according to the predicted results
    pose_eval.eval(toCameraCoord=args.toCameraCoord)
