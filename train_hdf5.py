# import open3d as o3d
import numpy as np
import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
import torch
import copy
import json
import os
from pathlib import Path
import pickle
import time
import re
import fire
import numpy as np
from functools import partial
import ast
import logging
from google.protobuf import text_format
import torchplus
from rslo.builder import voxel_builder
from rslo.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from rslo.protos import pipeline_pb2
from rslo.utils.log_tool import SimpleModelLog
from rslo.utils.progress_bar import ProgressBar
from collections import defaultdict
from rslo.builder import (
                                input_reader_builder,
                                 lr_scheduler_builder, 
                                 optimizer_builder,
                                 second_builder
                                 )

import torch.distributed as dist
from rslo.utils.distributed_utils import dist_init, average_gradients, DistModule, ParallelWrapper, DistributedSequatialSampler, DistributedGivenIterationSampler, DistributedGivenIterationSamplerEpoch, gradients_multiply
from torch.utils.data.distributed import DistributedSampler
from rslo.utils.util import modify_parameter_name_with_map
import random

GLOBAL_GPUS_PER_DEVICE = 1  # None
GLOBAL_STEP = 0
RANK=-1
WORLD_SIZE=-1

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    # global GLOBAL_GPUS_PER_DEVICE
    # device = device % GLOBAL_GPUS_PER_DEVICE or torch.device("cuda:0")
    # example_torch = defaultdict(list)

    example_torch = {}
    float_names = [
        "voxels", "lidar_seqs","normal_gt_seqs"
    ]
    for k, v in example.items():
        if k not in example_torch:
            example_torch[k] = []

        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            for i, _ in enumerate(v):
                if 1:#k !="lidar_seqs":
                    example_torch[k].append(torch.tensor(
                        v[i], dtype=torch.float32, device="cuda").to(dtype=dtype) )
                else:
                    example_torch[k].append(torch.tensor(
                        v[i], dtype=torch.float32, device="cuda") )
        elif k in ["tq_maps"]:
            example_torch[k] = [vi.to( device="cuda",dtype=dtype) for vi in v]
        elif k in ["odometry", "icp_odometry"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device="cuda").to(dtype=dtype)#.cuda()
        elif k in ["coordinates",  "num_points"]:
            for i, _ in enumerate(v):
                example_torch[k].append(torch.tensor(
                    v[i], dtype=torch.int32, device="cuda") )
        elif k in ['hier_points']:
            for t, _ in enumerate(v):
                example_torch[k].append([])
                for h,_ in enumerate(v[t]):
                    example_torch[k][t].append(torch.tensor(
                        v[t][h], dtype=torch.float32, device="cuda"))

        elif k == "num_voxels":
            for i, _ in enumerate(v):
                example_torch[k].append(torch.tensor(v[i]))
        else:
            for i, _ in enumerate(v):
                example_torch[k].append(v[i])
    return example_torch


def build_network(model_cfg, measure_time=False, testing=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    net = second_builder.build(
        model_cfg, voxel_generator,  measure_time=measure_time, testing=testing)
    return net


def _worker_init_fn(worker_id):
    # time_seed = np.array(time.time(), dtype=np.int32)
    global GLOBAL_STEP
    time_seed = GLOBAL_STEP
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])


def freeze_params_v2(params: dict, include: str = None, exclude: str = None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                p.requires_grad = False
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                p.requires_grad = False


def filter_param_dict(state_dict: dict, include: str = None, exclude: str = None):
    assert isinstance(state_dict, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    res_dict = {}
    for k, p in state_dict.items():
        if include_re is not None:
            if include_re.match(k) is None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is not None:
                continue
        res_dict[k] = p
    return res_dict


def chk_rank(rank_, use_dist=False):
    if not use_dist:
        return True
    global RANK
    if RANK<0:
        RANK=dist.get_rank()
    cur_rank = RANK#dist.get_rank()
    # self.world_size = dist.get_world_size()
    return cur_rank == rank_


def get_rank(use_dist=False):
    if not use_dist:
        return 0
    else:
        # return dist.get_rank()
        global RANK 
        if RANK<0:
            RANK=dist.get_rank()
        return RANK 


def get_world(use_dist):
    if not use_dist:
        return 1
    else:
        global WORLD_SIZE 
        if WORLD_SIZE<0:
            WORLD_SIZE=dist.get_world_size()
        return WORLD_SIZE #dist.get_world_size()
def get_ngpus_per_node():
    global GLOBAL_GPUS_PER_DEVICE
    return GLOBAL_GPUS_PER_DEVICE
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



def multi_proc_train(
          config_path,
          model_dir,
          use_apex,
          world_size,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pretrained_path=None,
          pretrained_include=None,
          pretrained_exclude=None,
          pretrained_param_map=None,
          freeze_include=None,
          freeze_exclude=None,
        #   multi_gpu=False,
          measure_time=False,
          resume=False,
          use_dist=False,
          gpus_per_node=1,
          start_gpu_id=0,
          optim_eval=False,
          seed=7,
          dist_port="23335",
        #   dist_url=None,
         force_resume_step=None,
         batch_size=None,
         apex_opt_level='O0'
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
          "pretrained_include": pretrained_include,
          "pretrained_exclude": pretrained_exclude,
          "pretrained_param_map": pretrained_param_map,
          "freeze_include": freeze_include,
          "freeze_exclude": freeze_exclude,
        #   "multi_gpu": multi_gpu,
          "measure_time": measure_time,
          "resume": resume,
          "use_dist": use_dist,
          "gpus_per_node": gpus_per_node,
          "optim_eval": optim_eval,
          "seed": seed,
          "dist_port": dist_port,
          "world_size": world_size,
          "force_resume_step":force_resume_step,
          "batch_size": batch_size,
          "apex_opt_level":apex_opt_level
        #   "dist_url": dist_url
    }
    from types import SimpleNamespace 
    params = SimpleNamespace(**params)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(x) for x in range(start_gpu_id, start_gpu_id+gpus_per_node))
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"  )

    mp.spawn(train_worker, nprocs=gpus_per_node,
                args=( params,) )

def train_worker(rank, params):
    global RANK, WORLD_SIZE
    RANK = rank
    WORLD_SIZE=params.world_size
    
    train(config_path=params.config_path,
          model_dir=params.model_dir,
          use_apex=params.use_apex,
          result_path=params.result_path,
          create_folder=params.create_folder,
          display_step=params.display_step,
        #   summary_step=params.summary_step,
          pretrained_path=params.pretrained_path,
          pretrained_include=params.pretrained_include,
          pretrained_exclude=params.pretrained_exclude,
          pretrained_param_map=params.pretrained_param_map,
          freeze_include=params.freeze_include,
          freeze_exclude=params.freeze_exclude,
        #   multi_gpu=params.multi_gpu,
          measure_time=params.measure_time,
          resume=params.resume,
          use_dist=params.use_dist,
          dist_port=params.dist_port,
          gpus_per_node=params.gpus_per_node,
          optim_eval=params.optim_eval,
          seed=params.seed,
          force_resume_step=params.force_resume_step,
          batch_size = params.batch_size,
          apex_opt_level=params.apex_opt_level
          ) 

#
def train(
         config_path,
          model_dir,
          use_apex,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          pretrained_path=None,
          pretrained_include=None,
          pretrained_exclude=None,
          pretrained_param_map=None,
          freeze_include=None,
          freeze_exclude=None,
          multi_gpu=False,
          measure_time=False,
          resume=False,
          use_dist=False,
          dist_port="23335",
          gpus_per_node=1,
          optim_eval=False,
          seed=7,
          force_resume_step=None,
          batch_size=None,
          apex_opt_level='O0'
          ):
    """train a VoxelNet model specified by a config file.
    """

    print("force_resume_step:", force_resume_step)
    print("torch.cuda.is_available()=", torch.cuda.is_available())
    print("torch.version.cuda=",torch.version.cuda) 
    dist_url=f"tcp://127.0.0.1:{dist_port}"
    print(f"dist_url={dist_url}", flush=True)
    # global RANK, WORLD_SIZE
    # RANK, WORLD_SIZE=rank, world_size
    global GLOBAL_GPUS_PER_DEVICE
    GLOBAL_GPUS_PER_DEVICE = gpus_per_node

  

    ######################################## initialize the distributed env #########################################
    if use_dist:
        # torch.cuda.set_device(get_rank(use_dist))
        if use_apex:
            dist.init_process_group(
                backend="nccl", init_method=dist_url, world_size=get_world(use_dist), rank=get_rank(use_dist))
        else:
            # rank, world_size = dist_init(str(dist_port))
            dist.init_process_group(
                backend="nccl", init_method=dist_url, world_size=get_world(use_dist), rank=get_rank(use_dist))
    
    print(get_rank(use_dist)%GLOBAL_GPUS_PER_DEVICE, flush=True)
    #set cuda device number
    torch.cuda.set_device(get_rank(use_dist)%GLOBAL_GPUS_PER_DEVICE)

    ############################################ create folders ############################################
    #fix random seeds
    # print(f"Set seed={seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    model_dir = str(Path(model_dir).resolve())

    model_dir = Path(model_dir)
    if chk_rank(0, use_dist):
        if not resume and model_dir.exists():
            raise ValueError("model dir exists and you don't specify resume.")
            print("Warning: model dir exists and you don't specify resume.")

        model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"

    ############################################# read config proto ############################################
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    if chk_rank(0, use_dist):
        with (model_dir / config_file_bkp).open("w+") as f:
            f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    eval_train_input_cfg = config.eval_train_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor


    ############################################# Update default options ############################################

    if batch_size is not None:
        input_cfg.batch_size = batch_size 

    ############################################# build network, optimizer etc. ############################################
    net = build_network(model_cfg, measure_time)  # .to(device)
    # print("Network is established!")
    net.cuda()
    fastai_optimizer = optimizer_builder.build(
        optimizer_cfg, net,
        mixed=False,
        loss_scale=loss_scale)

    voxel_generator = net.voxel_generator
    # print("num parameters:", len(list(net.parameters())), flush=True)

   

    ############################################# load pretrained model ############################################
    if pretrained_path is not None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        # print("Pretrained keys:", pretrained_dict.keys())
        # print("Model keys:", model_dict.keys()) 

        pretrained_dict = filter_param_dict(
            pretrained_dict, pretrained_include, pretrained_exclude)

        pretrained_dict = modify_parameter_name_with_map(
            pretrained_dict, ast.literal_eval(str(pretrained_param_map)))
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_pretrained_dict[k] = v
            else:
                print(k,v.shape)
        for k, v in model_dict.items():
            if not (k in pretrained_dict and v.shape == pretrained_dict[k].shape):
                # new_pretrained_dict[k] = v
                print("2",k,v.shape)
        # print("Load pretrained parameters:")
        # for k, v in new_pretrained_dict.items():
        #     print(k, v.shape)
        model_dict.update(new_pretrained_dict)
        net.load_state_dict(model_dict)
        freeze_params_v2(dict(net.named_parameters()),
                         freeze_include, freeze_exclude)
        net.clear_global_step()
        net.clear_metrics()
        del pretrained_dict

    ############################################# try to resume from the latest chkpt ############################################
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net] )
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [fastai_optimizer])
   
   
    ######################################## parallel the network  #########################################
    if use_dist:
        if use_apex:
            import apex
            apex.amp.register_float_function( torch, 'svd')
            apex.amp.register_float_function( torch, 'matmul')

            net, amp_optimizer = apex.amp.initialize(net.cuda(
                ), fastai_optimizer, opt_level=apex_opt_level, keep_batchnorm_fp32=None, loss_scale=None )
            net_parallel = apex.parallel.DistributedDataParallel(net)
        else:
            # net_parallel = ParallelWrapper(net.cuda(), 'dist')
            # amp_optimizer=fastai_optimizer
            net_parallel = torch.nn.parallel.DistributedDataParallel(net, device_ids=[get_rank()], find_unused_parameters=True)
            amp_optimizer = optimizer_builder.build(
                optimizer_cfg, net_parallel,
                mixed=False,
                loss_scale=loss_scale)

    else:
        net_parallel = net.cuda()

    ############################################# build lr_scheduler ############################################
    lr_scheduler=lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)
    
    
    if apex_opt_level in [ 'O2', 'O3']:#train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32


    ######################################## build dataloaders #########################################    
    # if multi_gpu:
    if use_dist:
        # num_gpu = dist.get_world_size()
        num_gpu = 1
        collate_fn = merge_second_batch
    else:
        num_gpu = torch.cuda.device_count()
        collate_fn = merge_second_batch_multigpu
    print(f"MULTI-GPU: use {num_gpu} gpu", flush=True)

    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        # target_assigner=target_assigner,
        multi_gpu=multi_gpu,
        use_dist=use_dist,
        split="train"
    )
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        use_dist=use_dist,
        split="val"
        # target_assigner=target_assigner
    )

    eval_train_dataset = input_reader_builder.build(
        eval_train_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        use_dist=use_dist,
        split="eval_train"
        # target_assigner=target_assigner
    )

    if use_dist:
        # train_sampler = DistributedSampler(dataset)
        # train_sampler = DistributedGivenIterationSampler(
        #     dataset, train_cfg.steps, input_cfg.batch_size, last_iter=net.get_global_step())
        train_sampler = DistributedGivenIterationSamplerEpoch(
            dataset, train_cfg.steps, input_cfg.batch_size, last_iter=net.get_global_step()-1, review_cycle=input_cfg.review_cycle)
        # train_sampler=DistributedSequatialSampler(dataset)
        shuffle = False
        eval_sampler = DistributedSequatialSampler(eval_dataset)
        eval_train_sampler = DistributedSequatialSampler(eval_train_dataset)

    else:
        train_sampler = None
        eval_sampler = None
        eval_train_sampler = None
        shuffle = True
        

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size * num_gpu ,
        shuffle=shuffle,
        num_workers=input_cfg.preprocess.num_workers * num_gpu,
        pin_memory=False,#True,#False,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=not multi_gpu,
        sampler=train_sampler
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size,  # only support multi-gpu train
        shuffle=False,
        sampler=eval_sampler,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)
    eval_train_dataloader = torch.utils.data.DataLoader(
        eval_train_dataset,
        batch_size=eval_train_input_cfg.batch_size,  # only support multi-gpu train
        shuffle=False,
        sampler=eval_train_sampler,
        num_workers=eval_train_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)
    ##########################################################################################   
    #                                            TRAINING
    ##########################################################################################   
    model_logging = SimpleModelLog(model_dir, disable=get_rank(use_dist) != 0)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag="config")
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    amp_optimizer.zero_grad()
    step_times = []
    step = start_step
    epoch = 0
    net_parallel.train()
    try:
        while True:
            if clear_metrics_every_epoch:
                net.clear_metrics()

            if use_dist:
                epoch = (net.get_global_step() *
                         input_cfg.batch_size) // len(dataloader)

                dataloader.sampler.set_epoch(epoch)
            else:
                epoch += 1
            for example in dataloader:
                
                global GLOBAL_STEP
                GLOBAL_STEP = step

               # all gpus load the same weights (from rank 1) 
                if net.freeze_bn and GLOBAL_STEP==force_resume_step:#GLOBAL_STEP==net.freeze_bn_start_step:
                    print("All gpus load the same weights (from rank 1)" )
                    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
                    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [amp_optimizer])
                    net.train()

                lr_scheduler.step(net.get_global_step())
                example_torch = example_convert_to_torch(
                    example, float_dtype, get_rank(use_dist))
  
                batch_size = example["voxels"][0].shape[0]
                ret_dict = net_parallel(example_torch)
                
                loss = ret_dict["loss"].mean()#/get_world(use_dist)
                translation_loss = ret_dict["translation_loss"].mean(
                )/get_world(use_dist)
                rotation_loss = ret_dict["rotation_loss"].mean(
                )/get_world(use_dist)
                pyramid_loss = ret_dict["pyramid_loss"].mean(
                )/get_world(use_dist)
               
                local_loss = ret_dict.get("local_loss", torch.Tensor([0]).cuda()).mean(
                )/get_world(use_dist)
                consistency_loss = ret_dict.get("C_loss", torch.Tensor([0]).cuda()).mean(
                )/get_world(use_dist)

                if ret_dict.get('local_features_loss', None) is not None:
                    local_features_loss = ret_dict.get('local_features_loss', None).mean(
                    )/get_world(use_dist)
                    reduced_local_features_loss = local_features_loss.data.clone()
                else:
                    reduced_local_features_loss= None
                reduced_loss = loss.data.clone()
                reduced_translation_loss = translation_loss.data.clone()
                reduced_rotation_loss = rotation_loss.data.clone()
                reduced_pyramid_loss = pyramid_loss.data.clone()
                reduced_local_loss = local_loss.data.clone()
                reduced_consistency_loss= consistency_loss.data.clone()
                if use_dist:
                    dist.all_reduce_multigpu(
                        [reduced_loss])
                    dist.all_reduce(reduced_translation_loss)
                    dist.all_reduce(reduced_rotation_loss)
                    dist.all_reduce(reduced_pyramid_loss)
                    dist.all_reduce(reduced_local_loss)
                    dist.all_reduce(reduced_consistency_loss)
                    if reduced_local_features_loss is not None:
                        dist.all_reduce(reduced_local_features_loss)

                amp_optimizer.zero_grad()
                if use_apex:
                    with apex.amp.scale_loss(loss, amp_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss = loss/get_world(use_dist)
                    loss.backward()
                    if use_dist:
                        average_gradients(net_parallel)

                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                amp_optimizer.step()

                net.update_global_step()
                step_time = (time.time() - t)
                step_times.append(step_time)
                t = time.time()
                metrics = defaultdict(dict)
                GLOBAL_STEP = net.get_global_step()

                if chk_rank(0, use_dist) and GLOBAL_STEP % display_step == 0:
                    print(f'Model directory: {str(model_dir)}')
                    if measure_time:
                        for name, val in net.get_avg_time_dict().items():
                            print(f"avg {name} time = {val * 1000:.3f} ms")

                    metrics["runtime"] = {
                        "step": GLOBAL_STEP,
                        "steptime": np.mean(step_times),
                    }
                    # metrics["runtime"].update(time_metrics[0])

                    metrics["loss"]["rotation_loss"] = float(
                        reduced_rotation_loss.detach().cpu().numpy())
                    metrics["loss"]["translation_loss"] = float(

                        reduced_translation_loss.detach().cpu().numpy())
                    metrics["loss"]["reduced_pyramid_loss"] = float(
                        reduced_pyramid_loss.detach().cpu().numpy())
                    metrics["loss"]["reduced_consistency_loss"] = float(
                        reduced_consistency_loss.detach().cpu().numpy())

                    metrics["parameters"]["translation_alpha"] = float(
                        net._translation_loss.alpha.detach().cpu().numpy())
                    metrics["parameters"]["rotation_alpha"] = float(
                        net._rotation_loss.alpha.detach().cpu().numpy())
                    metrics["parameters"]["py_translation_alpha"] = float(
                        net._pyramid_translation_loss.alpha.detach().cpu().numpy())
                    metrics["parameters"]["py_rotation_alpha"] = float(
                        net._pyramid_rotation_loss.alpha.detach().cpu().numpy())
                    try:
                        metrics["parameters"]["consistency_alpha"] = float(
                        net._consistency_loss.alpha.detach().cpu().numpy())
                    except:
                        pass

                    metrics['learning_rate'] = amp_optimizer.lr
                    metrics['epoch'] = epoch
                    # metrics["parameters"]['dynamic_sigma'] = float(
                    #     net.odom_predictor.dynamic_sigma.param.detach().cpu().numpy())
                    

                    model_logging.log_metrics(metrics, GLOBAL_STEP)
                #
                if optim_eval and GLOBAL_STEP<total_step/2 and GLOBAL_STEP>train_cfg.steps_per_eval*2:
                    steps_per_eval = 2*train_cfg.steps_per_eval
                else:
                    steps_per_eval = train_cfg.steps_per_eval
                
                if GLOBAL_STEP % steps_per_eval == 0:
                    if chk_rank(0, use_dist):  # logging
                        # torchplus.train.save_models_cpu(model_dir, [net, amp_optimizer],
                        #                                 net.get_global_step())
                        torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                        net.get_global_step())

                        #summarize BN statistics
                        running_mean = []
                        running_var = []
                        bn_weights=[]
                        bn_bias=[]
                        bn_mean_dyn_mom=[]
                        bn_var_dyn_mom=[]

                        # model_logging.log_images(
                        #     {'middle_feature_a': ret_dict['middle_feature'][0].cpu().detach()}, GLOBAL_STEP, prefix='a')
                        # model_logging.log_images(
                        #     {'middle_feature_b': ret_dict['middle_feature'][-1].cpu().detach()}, GLOBAL_STEP, prefix='b')
                        model_logging.log_images(
                            {'mask': ret_dict['feature_mask'].cpu().detach()}, GLOBAL_STEP, prefix='')
                        if ret_dict['t_conf'] is not None:
                            model_logging.log_images(
                                {'t_conf': (ret_dict['t_conf']/torch.max(ret_dict['t_conf'])).cpu().detach() }, GLOBAL_STEP, prefix='')
                            model_logging.log_images(
                                {'r_conf': (ret_dict['r_conf']/torch.max(ret_dict['r_conf'])).cpu().detach() }, GLOBAL_STEP, prefix='')
                        if example.get('tq_maps', None) is not None and  not isinstance(example.get('tq_maps', None), (list, tuple)):
                            
                            tq_maps_gt = example['tq_maps'].view(
                                -1, *example['tq_maps'].shape[2:])[:, :3]
                            model_logging.log_images(
                                {'t_map_gt': ((torch.nn.functional.normalize(tq_maps_gt) + 1)/2).cpu().detach() }, GLOBAL_STEP, prefix='')
                        
                        if ret_dict.get('tq_map_g', None) is not None:
                            tq_map_g = ret_dict.get('tq_map_g', None)
                    
                            model_logging.log_images(
                                {'t_map_g': ((torch.nn.functional.normalize(tq_map_g[:,:3]) + 1)/2).cpu().detach() }, GLOBAL_STEP, prefix='')
                            model_logging.log_images(
                                {'q_map_g': ((torch.nn.functional.normalize(tq_map_g[:,4:7]) + 1)/2).cpu().detach() }, GLOBAL_STEP, prefix='')

                            if(len(example_torch['odometry'].shape)==3):
                                example_torch['odometry']=example_torch['odometry'].squeeze(dim=0)
                            T_targets = example_torch['odometry'][...,:3].reshape(-1,3,1,1)
                            R_targets = example_torch['odometry'][...,3:].reshape(-1,4,1,1)[:,1:]

                            model_logging.log_images(
                                {'t_map_g_gt': ((torch.nn.functional.normalize(T_targets.expand_as(tq_map_g[:,:3])) + 1)/2).cpu().detach() }, GLOBAL_STEP, prefix='')
                            model_logging.log_images(
                                {'q_map_g_gt': ((torch.nn.functional.normalize(R_targets.expand_as(tq_map_g[:,3:6]) ) + 1)/2).cpu().detach()}, GLOBAL_STEP, prefix='')

                        if ret_dict.get('pyramid_motion', None) is not None:
                            pyramid_motions = ret_dict['pyramid_motion']
                            for i, (p, m) in enumerate(pyramid_motions):
                                model_logging.log_images(
                                    {f'pyramid_motion_pred{i}': ((torch.nn.functional.normalize(p[:, :3]) + 1)/2).cpu().detach() }, GLOBAL_STEP, prefix='')
                                model_logging.log_images(
                                    {f'pyramid_motion_mast{i}': (m/m.max()).mean(dim=1, keepdim=True).cpu().detach() }, GLOBAL_STEP, prefix='')

                    del ret_dict
                    eval_once(net,
                              eval_dataset=eval_dataset, eval_dataloader=eval_dataloader, eval_input_cfg=eval_input_cfg,
                              result_path=result_path,
                              global_step=GLOBAL_STEP,
                              model_logging=model_logging,
                              metrics=metrics,
                              float_dtype=float_dtype,
                              use_dist=use_dist,
                              prefix='eval_')
                    eval_once(net,
                                eval_dataset=eval_train_dataset, eval_dataloader=eval_train_dataloader, eval_input_cfg=eval_train_input_cfg,
                                result_path=result_path,
                                global_step=GLOBAL_STEP,
                                model_logging=model_logging,
                                metrics=metrics,
                                float_dtype=float_dtype,
                                use_dist=use_dist,
                                prefix='eval_train_')
                    net.train()
                
                step += 1
                if step >= total_step:
                    break
            if step >= total_step:
                break
    except Exception as e:
        model_logging.log_text(str(e), step)
        raise e
    finally:
        model_logging.close()
    # torchplus.train.save_models_cpu(model_dir, [net, amp_optimizer],
    #                                 net.get_global_step())
    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                        net.get_global_step())


def eval_once(net,
              eval_dataset, eval_dataloader, eval_input_cfg,
              result_path, global_step, model_logging, metrics, float_dtype, use_dist, prefix='eval_'):
    import rslo.utils.pose_utils as tch_p 
    net.eval()
    result_path_step = result_path / \
            f"step_{global_step}"
    if chk_rank(0, use_dist):
        result_path_step.mkdir(parents=True, exist_ok=True)
        model_logging.log_text("#################################",
                               global_step)
        model_logging.log_text("# EVAL", global_step)
        model_logging.log_text("#################################",
                               global_step)
        model_logging.log_text(
            "Generate output labels...", global_step)
        prog_bar = ProgressBar()
        prog_bar.start(len(eval_dataloader))
    t = 0
    results = []

    cnt = 0  
    for i, example in enumerate(eval_dataloader):
        example = example_convert_to_torch(
            example, float_dtype, device=get_rank(use_dist))
        results.append(net(example))

        if results[-1]['rotation_preds'].shape[-1]==3: #log_quaternion
            results[-1]['rotation_preds'] = tch_p.qexp_t(results[-1]['rotation_preds'])
        if chk_rank(0, use_dist) and i%10==0:
            prog_bar.print_bar(finished_size=10)

    if use_dist:  # chk_rank(0, use_dist):
         
        results = [torch.cat(
            [d['translation_preds'], d['rotation_preds']], dim=1) for d in results]
        results = torch.cat(results, dim=0)

        if 1:  # chk_rank(0, use_dist):
            gather_list = [torch.zeros_like(results)
                           for i in range(get_world(use_dist))]
            dist.all_gather(gather_list, results)
            
            results = torch.cat(
                gather_list, dim=-1).reshape([-1, 7])[:len(eval_dataset)]

    if chk_rank(0, use_dist):
        result_dict = eval_dataset.dataset.evaluation_seqs(
            results, str(result_path_step))

        for k, v in result_dict['error']['kitti_error'].items():
            if 1:  # len(v) > 1:
                metrics[f'{prefix}{k}'] = v
                metrics[f'{prefix}{k}'] = v
            else:
                metrics[f'{prefix}{k}'] = -1
    if chk_rank(0, use_dist):
        model_logging.log_metrics(metrics, global_step)
        model_logging.log_images(
            result_dict['plot']['odometry_pred'], global_step, prefix=prefix)
    dist.barrier()
    del results
    net.train()


if __name__ == '__main__':

    fire.Fire()
