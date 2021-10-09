import torchplus
import random
import apex
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from rslo.utils.util import modify_parameter_name_with_map
from torch.utils.data.distributed import DistributedSampler
from rslo.utils.distributed_utils import dist_init, average_gradients, DistModule, ParallelWrapper, DistributedSequatialSampler, DistributedGivenIterationSampler, DistributedGivenIterationSamplerEpoch
import torch.distributed as dist
from collections import defaultdict
from rslo.utils.progress_bar import ProgressBar
from rslo.utils.log_tool import SimpleModelLog
from rslo.builder import (input_reader_builder,
                          lr_scheduler_builder, optimizer_builder,
                          second_builder)
from rslo.protos import pipeline_pb2
from rslo.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from rslo.builder import voxel_builder
from google.protobuf import text_format
from functools import partial
import torch
import numpy as np
import fire
import re
import time
import shutil
import pickle
from pathlib import Path
import os
import json
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# import rslo.data.kitti_common as kitti
# import psutil

# from post_refinement.src.vote_refine import *
global_gpus_per_device = 1  # None
global_step = 0
# GLOBAL_GPUS_PER_DEVICE = 1  # None
RANK = -1
WORLD_SIZE = -1


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    global global_gpus_per_device
    device = device % global_gpus_per_device or torch.device("cuda:0")
    # example_torch = defaultdict(list)
    example_torch = {}
    float_names = [
        "voxels",
    ]
    # import pdb
    # pdb.set_trace()
    for k, v in example.items():
        if k not in example_torch:
            example_torch[k] = []

        if k in float_names:
            for i, _ in enumerate(v):
                example_torch[k].append(torch.tensor(
                    v[i], dtype=torch.float32, device=device).to(dtype))
        elif k in ["tq_maps"]:
            example_torch[k] = [vi.to(device=device, dtype=dtype) for vi in v]
        elif k in ["odometry", ]:

            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates",  "num_points"]:
            for i, _ in enumerate(v):
                example_torch[k].append(torch.tensor(
                    v[i], dtype=torch.int32, device=device))

        # elif k == "calib":
        #     calib = {}
        #     for k1, v1 in v.items():
        #         calib[k1] = torch.tensor(
        #             v1, dtype=dtype, device=device).to(dtype)
        #     example_torch[k] = calib
        elif k == "num_voxels":
            for i, _ in enumerate(v):
                example_torch[k].append(torch.tensor(v[i]))
        elif k == 'lidar_seqs':
            example_torch[k] = [[torch.tensor(v[i][j], dtype=torch.float32, device=device) for j in range(
                len(v[i]))] for i in range(len(v))]
        elif k in ['hier_points']:
            for t, _ in enumerate(v):
                example_torch[k].append([])
                for h, _ in enumerate(v[t]):
                    example_torch[k][t].append(torch.tensor(
                        v[t][h], dtype=torch.float32, device="cuda"))

        else:
            # example_torch[k] = v
            for i, _ in enumerate(v):
                example_torch[k].append(v[i])
    return example_torch


def build_network(model_cfg, measure_time=False, testing=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    net = second_builder.build(
        model_cfg, voxel_generator,  measure_time=measure_time, testing=testing)
    return net


def chk_rank(rank_, use_dist=False):
    if not use_dist:
        return True
    global RANK
    if RANK < 0:
        RANK = dist.get_rank()
    cur_rank = RANK  # dist.get_rank()
    return cur_rank == rank_


def get_rank(use_dist=False):
    if not use_dist:
        return 0
    else:
        # return dist.get_rank()
        global RANK
        if RANK < 0:
            RANK = dist.get_rank()
        return RANK


def get_world(use_dist):
    if not use_dist:
        return 1
    else:
        global WORLD_SIZE
        if WORLD_SIZE < 0:
            WORLD_SIZE = dist.get_world_size()
        return WORLD_SIZE  # dist.get_world_size()


def get_ngpus_per_node():
    global GLOBAL_GPUS_PER_DEVICE
    return GLOBAL_GPUS_PER_DEVICE


def prediction_filter(pred_dict, include, exclude=[]):
    for k in list(pred_dict.keys()):
        if k not in include or k in exclude:
            pred_dict.pop(k)
    return pred_dict


def multi_proc_eval(
    config_path,
    model_dir,
    use_apex,
    world_size,
    result_path=None,
    create_folder=False,
    display_step=50,
    summary_step=5,
    pretrained_path=None,
    use_dist=False,
    gpus_per_node=1,
    dist_port="23335",
    refine=False,
    output_dir=None,
    visible_gpus=None,
):

    params = {
        "config_path": config_path,
        "model_dir": model_dir,
        "use_apex": use_apex,
        "result_path": result_path,
        "create_folder": create_folder,
        "display_step": display_step,
        "summary_step": summary_step,
        "pretrained_path": pretrained_path,
        "use_dist": use_dist,
        "gpus_per_node": gpus_per_node,
        "dist_port": dist_port,
        "world_size": world_size,
        "refine": refine,
        "output_dir": output_dir,
        #   "dist_url": dist_url
    }
    from types import SimpleNamespace
    params = SimpleNamespace(**params)

    if visible_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            str(x) for x in range(gpus_per_node))
    else:
        print(visible_gpus, flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            str(x) for x in visible_gpus)
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    mp.spawn(eval_worker, nprocs=gpus_per_node,
             args=(params,))
    # else:
    #     main_worker(args.train_gpu, args.ngpus_per_node, args)


def eval_worker(rank, params):
    global RANK, WORLD_SIZE
    RANK = rank
    WORLD_SIZE = params.world_size

    evaluate(config_path=params.config_path,
             model_dir=params.model_dir,
             use_apex=params.use_apex,
             result_path=params.result_path,
             pretrained_path=params.pretrained_path,
             use_dist=params.use_dist,
             dist_port=params.dist_port,
             gpus_per_node=params.gpus_per_node,
             refine=params.refine,
             output_dir=params.output_dir
             )


def evaluate(config_path,
             model_dir=None,
             result_path=None,
             pretrained_path=None,
             measure_time=True,
             batch_size=None,
             gpus_per_node=1,
             use_dist=False,
             dist_port="23335",
             use_apex=False,
             use_cyc_merge=False,
             refine=False,
             seed=0,
             output_dir=None,
             **kwargs):

    from rslo.utils.pose_utils import invert_pose_quaternion, compose_pose_quaternion
    if use_cyc_merge:
        assert(not use_dist)
    global global_gpus_per_device
    global_gpus_per_device = gpus_per_node

    dist_url = f"tcp://127.0.0.1:{dist_port}"

    ######################################## initialize the distributed env #########################################
    if use_dist:
        if use_apex:
            dist.init_process_group(
                backend="nccl", init_method=dist_url, world_size=get_world(use_dist), rank=get_rank(use_dist))
        else:
            rank, world_size = dist_init(str(dist_port))
    # set cuda device number
    torch.cuda.set_device(get_rank(use_dist) % global_gpus_per_device)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_logging = SimpleModelLog(model_dir, disable=get_rank(use_dist) != 0)
    model_logging.open()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_name = 'eval_results'

    if result_path is None:
        model_dir = Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = Path(result_path)
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
            print(proto_str)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time=measure_time,
                        testing=True).cuda()  # .to(device)

    ######################################## parallel the network  #########################################
    if use_dist:
        if use_apex:
            import apex
            net = apex.amp.initialize(net.cuda(
            ),  opt_level="O0", keep_batchnorm_fp32=None, loss_scale=None)
            net_parallel = apex.parallel.DistributedDataParallel(net)
        else:
            net_parallel = ParallelWrapper(net.cuda(), 'dist')
            # amp_optimizer=fastai_optimizer

    else:
        net_parallel = net.cuda()

    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    voxel_generator = net.voxel_generator

    if pretrained_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:

        torchplus.train.restore(pretrained_path, net, )
    batch_size = batch_size or input_cfg.batch_size
    if use_cyc_merge:
        assert batch_size == 1
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        split='val'  # 'train' #'val'
    )
    if use_dist:
        eval_sampler = DistributedSequatialSampler(eval_dataset)
    else:
        eval_sampler = None

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=input_cfg.preprocess.num_workers,
        pin_memory=False,
        sampler=eval_sampler,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    results = []
    bar = ProgressBar()
    # bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    bar.start(len(eval_dataloader))
    prep_example_times = []
    prep_times = []
    fwd_times = []
    cluster_times = []

    t2 = time.time()

    cnt = 0
    for example in iter(eval_dataloader):
        if measure_time:
            prep_times.append(time.time() - t2)
            torch.cuda.synchronize()
            t1 = time.time()
        example = example_convert_to_torch(
            example, float_dtype, get_rank(use_dist))

        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)
            t1 = time.time()

        with torch.no_grad():
            pred = net_parallel(example)
            new_points = pred['voxel_features']

            # print(type(new_points), len(new_points), new_points[0].shape )
            # print (pred['translation_preds'].shape)
            if measure_time:
                fwd_times.append(time.time()-t1)
                t1 = time.time()

            result = torch.cat(
                [pred['translation_preds'], pred['rotation_preds']], dim=1)

            results.append(result)

        if chk_rank(0, use_dist):
            bar.print_bar()
            print('', end='\r', flush=True)
        if measure_time:
            t2 = time.time()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    if measure_time:
        print(
            f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms"
        )
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
        print(f"avg forward time: {np.mean(fwd_times) * 1000:.3f} ms")
        print(f"avg cluster time: {np.mean(cluster_times) * 1000:.3f} ms")

    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")

    if use_dist:  # chk_rank(0, use_dist):
        results = torch.cat(results, dim=0)
        dist.barrier()
        if 1:  # chk_rank(0, use_dist):
            gather_list = [torch.zeros_like(results)
                           for i in range(get_world(use_dist))]

            dist.all_gather(gather_list, results)

            results = torch.cat(
                gather_list, dim=-1).reshape([-1, 7])[:len(eval_dataset)]

    if chk_rank(0, use_dist):
        result_dict = eval_dataset.dataset.evaluation_seqs(
            results, str(result_path_step))

        metrics = defaultdict(dict)

        prefix = 'test_'
        for k, v in result_dict['error']['kitti_error'].items():
            metrics[f'{prefix}{k}'] = v
            metrics[f'{prefix}{k}'] = v
        model_logging.log_metrics(metrics, global_step)
        # model_logging.log_images(
        #     result_dict['plot']['odometry_pred'], global_step, prefix=prefix)


if __name__ == '__main__':
    fire.Fire()
