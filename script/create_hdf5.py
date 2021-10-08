import open3d as o3d
import os
import numpy as np
import copy
import pickle
import fire

import h5py
def parse_pose_file(file):
    with open(file) as f:
        lines = [line.strip() for line in f.readlines()]

    poses = []
    for line in lines:
        poses.append(
            np.array([np.float32(l) for l in line.split()],
                     dtype=np.float32).reshape((3, 4))
        )
    return poses


def parse_calib_file(file):
    info = {}
    with open(file) as f:
        lines = [line.strip() for line in f.readlines()]
    for i, l in enumerate(lines):
        nums = np.array([np.float32(x)
                         for x in l.split(' ')[1:]], dtype=np.float32)
        if i < 4:
            info[f"calib/P{i}"] = nums.reshape((3, 4))
        else:
            info[f"calib/Tr_velo_to_cam"] = nums.reshape((3, 4))
    return info


# def create_data_info(data_root, saving_path, is_test_data=False):
def create_data_info(data_root, saving_path, data_type='train'):
    """[summary]
        info structure:
        {
            0:[
                {
                "index": idx,
                "lidar_bin_path": lidar_bin_paths[idx],
                "pose": poses[idx],
                "calib/P0": None,
                "calib/P1": None,
                "calib/P2": None,
                "calib/P3": None,
                "calib/Tr_velo_to_cam": None
                },
                {
                "index": idx,
                "lidar_bin_path": lidar_bin_paths[idx],
                "pose": poses[idx],
                "calib/P0": None,
                "calib/P1": None,
                "calib/P2": None,
                "calib/P3": None,
                "calib/Tr_velo_to_cam": None
            }
            ...
            ],
            1:[

            ]
            ...
        }

    """
    assert data_type in ['train', 'eval', 'test', 'eval_train', 'local']

    lidar_dir = os.path.join(data_root, 'sequences')
    pose_dir = os.path.join(data_root, 'poses')
    if data_type == 'train':
        # seqs = [f'{i:02d}' for i in range(9)]
        seqs = [f'{i:02d}' for i in range(7)]
    elif data_type == 'eval':
        #seqs = [f'{i:02d}' for i in range(9, 11)]
        seqs = [f'{i:02d}' for i in range(7, 11)]
        # seqs = [f'{i:02d}' for i in range(8, 9)]
    elif data_type == 'test':
        seqs = [f'{i:02d}' for i in range(11, 22)]
    elif data_type == 'eval_train':
        seqs = [f'{i:02d}' for i in range(0, 1)]
    elif data_type == "local":
        seqs = [f'{i:02d}' for i in[0,8]]

    # create training data
    res = {}
    for seq in seqs:
        res[int(seq)] = []

        lidar_bin_dir = os.path.join(lidar_dir, seq, 'velodyne')
        lidar_bin_paths = os.listdir(lidar_bin_dir)
        lidar_bin_paths = [os.path.join('sequences', seq, 'velodyne', p)
                           for p in lidar_bin_paths]
        lidar_bin_paths.sort()

        # read
        lidar_calib_path = os.path.join(lidar_dir, seq, 'calib.txt')
        calib = parse_calib_file(lidar_calib_path)

        if data_type != 'test':
            lidar_pose_path = os.path.join(pose_dir, f"{seq}.txt")
            poses = parse_pose_file(lidar_pose_path)
        else:
            poses = [None]*len(lidar_bin_paths)

        for idx in range(len(lidar_bin_paths)):
            info = {
                "index": idx,
                "lidar_bin_path": lidar_bin_paths[idx],
                "pose": poses[idx],
                "calib/P0": None,
                "calib/P1": None,
                "calib/P2": None,
                "calib/P3": None,
                "calib/Tr_velo_to_cam": None
            }
            info.update(calib)

            res[int(seq)].append(info)

    with open(saving_path, 'wb+') as f:

        print("Total data amount:", np.sum([len(res[r]) for r in res]))
        pickle.dump(res, f)

def estimate_normal(points, radius=0.6, max_nn=30, camera_location=(0,0,0) ):
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
def create_data_info_hdf5(f, data_root, data_type='all'):
    """[summary]
        info structure:
        {
            0:[
                {
                "index": idx,
                "lidar_bin_path": lidar_bin_paths[idx],
                "pose": poses[idx],
                "calib/P0": None,
                "calib/P1": None,
                "calib/P2": None,
                "calib/P3": None,
                "calib/Tr_velo_to_cam": None
                },
                {
                "index": idx,
                "lidar_bin_path": lidar_bin_paths[idx],
                "pose": poses[idx],
                "calib/P0": None,
                "calib/P1": None,
                "calib/P2": None,
                "calib/P3": None,
                "calib/Tr_velo_to_cam": None
            }
            ...
            ],
            1:[

            ]
            ...
        }

    """
    assert data_type in ['train', 'eval', 'test', 'eval_train', 'local','all']

    lidar_dir = os.path.join(data_root, 'sequences')
    pose_dir = os.path.join(data_root, 'poses')
    if data_type == 'train':
        # seqs = [f'{i:02d}' for i in range(9)]
        seqs = [f'{i:02d}' for i in range(7)]
    elif data_type == 'eval':
        #seqs = [f'{i:02d}' for i in range(9, 11)]
        seqs = [f'{i:02d}' for i in range(7, 11)]
        # seqs = [f'{i:02d}' for i in range(8, 9)]
    elif data_type == 'test':
        seqs = [f'{i:02d}' for i in range(11, 22)]
    elif data_type == 'eval_train':
        seqs = [f'{i:02d}' for i in range(0, 1)]
    elif data_type == "local":
        seqs = [f'{i:02d}' for i in[0,8]]
    elif data_type == "all":
        # seqs = [f'{i:02d}' for i in range(11) ]
        seqs = [f'{i:02d}' for i in range(22) ]

    print(seqs)
    # grp = f.create_group(data_type) 

    # # hdf5_file = h5py.File('yourdataset.hdf5', mode='w')
    # dt = h5py.special_dtype(vlen=np.dtype('float32'))
    # # hdf5_file.create_dataset('dataset', (3,), dtype=dt)
    # # hdf5_file['dataset'][...] = arrs

    # # for seq_grp in [train_grp, eval_grp, train_eval_grp]:
    # # seq_grp.create_dataset("lidar_points", (100, 10, 4), np.float32, chunks=(1, 200000, 4), maxshape=(None,None,4) ) #LxNxC
    # # seq_grp.create_dataset("lidar_normals", (100, 10, 3), np.float32, chunks=(1, 200000, 3), maxshape=(None,None,3) ) #LxNxC
    # seq_grp.create_dataset("lidar_points", (100, ), dtype=dt, chunks=(1, ), maxshape=(None,) ) #Lx(Nx4)
    # seq_grp.create_dataset("lidar_normals", (100,), dtype=dt, chunks=(1, ), maxshape=(None,) ) #Lx(Nx3)


    # # seq_grp.create_dataset("valid_point_cnt", (100, ), np.int32, chunks=True, maxshape=(None,) ) #L

    # # seq_grp.create_dataset("hier_lidar_points_normals_0.1", (100,10, 6), np.float32, chunks=(1, 100000, 6), maxshape=(None,None,6) ) #LxNxC
    # # seq_grp.create_dataset("hier_valid_point_cnt_0.1", (100, ), np.int32, chunks=True, maxshape=(None,) ) #L
    # seq_grp.create_dataset("hier_lidar_points_normals_0.1", (100, ), dtype=dt, chunks=(1,), maxshape=(None,) ) #Lx(NxC)

    # seq_grp.create_dataset("poses", (100, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
    # seq_grp.create_dataset("calib.P0", (100, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
    # seq_grp.create_dataset("calib.P1", (100, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
    # seq_grp.create_dataset("calib.P2", (100, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
    # seq_grp.create_dataset("calib.P3", (100, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
    # seq_grp.create_dataset("calib.Tr_velo_to_cam", (100, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC


    # create training data
    res = {}
    # acc=0
    for seq in seqs:
        res[int(seq)] = []

        lidar_bin_dir = os.path.join(lidar_dir, seq, 'velodyne')
        lidar_bin_paths = os.listdir(lidar_bin_dir)
        lidar_bin_paths = [os.path.join('sequences', seq, 'velodyne', p)
                           for p in lidar_bin_paths]
        lidar_bin_paths.sort()
        seq_len=len(lidar_bin_paths)


        # read
        lidar_calib_path = os.path.join(lidar_dir, seq, 'calib.txt')
        calib = parse_calib_file(lidar_calib_path)

        if data_type != 'test' and int(seq) <11:
            lidar_pose_path = os.path.join(pose_dir, f"{seq}.txt")
            poses = parse_pose_file(lidar_pose_path)
        else:
            poses = [np.zeros([3,4], dtype=np.float32)]*len(lidar_bin_paths)

        seq_grp = f.create_group(seq)
        #resize dataset
        # hdf5_file = h5py.File('yourdataset.hdf5', mode='w')
        dt = h5py.special_dtype(vlen=np.dtype('float32'))
        seq_grp.create_dataset("lidar_points", (seq_len, ), dtype=dt, chunks=(3, ), maxshape=(None,) ) #Lx(Nx4)
        seq_grp.create_dataset("lidar_normals", (seq_len,), dtype=dt, chunks=(3, ), maxshape=(None,) ) #Lx(Nx3)
        seq_grp.create_dataset("hier_lidar_points_normals_0.1", (seq_len, ), dtype=dt, chunks=(3,), maxshape=(None,) ) #Lx(NxC)
        seq_grp.create_dataset("hier_lidar_points_normals_0.2", (seq_len, ), dtype=dt, chunks=(3,), maxshape=(None,) ) #Lx(NxC)
        seq_grp.create_dataset("hier_lidar_points_normals_0.4", (seq_len, ), dtype=dt, chunks=(3,), maxshape=(None,) ) #Lx(NxC)
        seq_grp.create_dataset("hier_lidar_points_normals_0.8", (seq_len, ), dtype=dt, chunks=(3,), maxshape=(None,) ) #Lx(NxC)
        seq_grp.create_dataset("poses", (seq_len, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
        seq_grp.create_dataset("calib.P0", (seq_len, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
        seq_grp.create_dataset("calib.P1", (seq_len, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
        seq_grp.create_dataset("calib.P2", (seq_len, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
        seq_grp.create_dataset("calib.P3", (seq_len, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC
        seq_grp.create_dataset("calib.Tr_velo_to_cam", (seq_len, 3, 4), np.float32, chunks=True, maxshape=(None,3,4) ) #LxNxC

        # for k in seq_grp.keys():
        #     print(k, acc, flush=True)
        #     if len(seq_grp[k])<acc:
        #         seq_grp[k].resize(acc, axis=0)


        for idx in range(len(lidar_bin_paths)):
            info = {
                "index": idx,
                "lidar_bin_path": lidar_bin_paths[idx],
                "pose": poses[idx],
                "calib/P0": None,
                "calib/P1": None,
                "calib/P2": None,
                "calib/P3": None,
                "calib/Tr_velo_to_cam": None
            }
            info.update(calib)

            res[int(seq)].append(info)

            #write hdf5 

            #lidar points
            lidar_path = os.path.join(data_root,  info["lidar_bin_path"])
            lidar_points =  np.fromfile(
                    lidar_path, dtype=np.float32,count=-1).reshape([-1, 4])  
            
            if np.count_nonzero(np.isnan(lidar_points))>0:
                nan_mask=~np.isnan(lidar_points.sum(axis=-1) )
                lidar_points = lidar_points[nan_mask]
                print("Warning: NaN exists in lidar points! After cleaning NaNs, skipped!")
                continue

            # seq_grp["valid_point_cnt"][idx] = len(lidar_points)
            # seq_grp["lidar_points"][idx, :seq_grp["valid_point_cnt"][idx]] = lidar_points 
            # seq_grp["lidar_points"][idx] = lidar_points.reshape(-1)

            # try:
            #normal
            lidar_pcd = estimate_normal(lidar_points[:,:3])
            lidar_normal = np.asarray(lidar_pcd.normals)
            # seq_grp["lidar_normals"][idx, :seq_grp["valid_point_cnt"][idx]] = lidar_normal 
            seq_grp["lidar_normals"][idx] = lidar_normal.reshape(-1) 

            #hierarchical
            lidar_pcd_10 = down_sample_points(lidar_pcd, 0.1) 
            seq_grp["hier_lidar_points_normals_0.1"][idx] = np.concatenate([np.asarray(lidar_pcd_10.points), np.asarray(lidar_pcd_10.normals)], axis=-1).reshape(-1) #Lx6

            lidar_pcd_20 = down_sample_points(lidar_pcd, 0.2) 
            seq_grp["hier_lidar_points_normals_0.2"][idx] = np.concatenate([np.asarray(lidar_pcd_20.points), np.asarray(lidar_pcd_20.normals)], axis=-1).reshape(-1) #Lx6

            lidar_pcd_40 = down_sample_points(lidar_pcd, 0.4) 
            seq_grp["hier_lidar_points_normals_0.4"][idx] = np.concatenate([np.asarray(lidar_pcd_40.points), np.asarray(lidar_pcd_40.normals)], axis=-1).reshape(-1) #Lx6

            lidar_pcd_80 = down_sample_points(lidar_pcd, 0.8) 
            seq_grp["hier_lidar_points_normals_0.8"][idx] = np.concatenate([np.asarray(lidar_pcd_80.points), np.asarray(lidar_pcd_80.normals)], axis=-1).reshape(-1) #Lx6
            print(lidar_path,  lidar_points.shape,lidar_normal.shape, lidar_normal[0], lidar_pcd, flush=True)
            # except:
            #     print("Warning: lidar points containing weird values, skipped!")
            #     continue

            seq_grp["lidar_points"][idx] = lidar_points.reshape(-1)
            #calib
            seq_grp["poses"][idx] = info["pose"]
            seq_grp["calib.P0"][idx] = info["calib/P0"]
            seq_grp["calib.P1"][idx] = info["calib/P1"]
            seq_grp["calib.P2"][idx] = info["calib/P2"]
            seq_grp["calib.P3"][idx] = info["calib/P3"]
            seq_grp["calib.Tr_velo_to_cam"][idx] = info["calib/Tr_velo_to_cam"]
            



def create_hdf5(data_root, hdf5_path, data_type='all'):
    f = h5py.File(hdf5_path, mode='w', rdcc_nbytes=1024**2*15) #15MB

    create_data_info_hdf5(f, data_root, data_type=data_type)
    # create_data_info_hdf5(f, data_root, data_type='eval')
    # create_data_info_hdf5(f, data_root, data_type='eval_train')

    f.close()



if __name__ == '__main__':
    fire.Fire()
