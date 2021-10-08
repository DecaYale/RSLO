# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Input reader builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""

from rslo.protos import input_reader_pb2
from rslo.data.dataset import get_dataset_class
from rslo.data.preprocess import prep_pointcloud
import numpy as np
from functools import partial
# from rslo.utils.config_tool import get_downsample_factor


def build(input_reader_config,
          model_config,
          training,
          voxel_generator,
          multi_gpu=False,
          use_dist=False,
          split=None,
          use_hdf5=False
          ):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    prep_cfg = input_reader_config.preprocess
    dataset_cfg = input_reader_config.dataset
    num_point_features = model_config.num_point_features
    cfg = input_reader_config
    db_sampler = None
    # if len(db_sampler_cfg.sample_groups) > 0 or db_sampler_cfg.database_info_path != "":  # enable sample
    #     db_sampler = dbsampler_builder.build(db_sampler_cfg)
    grid_size = voxel_generator.grid_size
    # feature_map_size = grid_size[:2] // out_size_factor
    # feature_map_size = [*feature_map_size, 1][::-1]

    dataset_cls = get_dataset_class(dataset_cfg.dataset_class_name)
    # assert dataset_cls.NumPointFeatures >= 3, "you must set this to correct value"
    # assert dataset_cls.NumPointFeatures == num_point_features, "currently you need keep them same"

    prep_func = partial(
        prep_pointcloud,
        root_path=dataset_cfg.kitti_root_path,
        voxel_generator=voxel_generator,
        # target_assigner=target_assigner,
        training=training,
        max_voxels=prep_cfg.max_number_of_voxels,
        shuffle_points=prep_cfg.shuffle_points,

        num_point_features=num_point_features,  # dataset_cls.NumPointFeatures,

        # out_size_factor=out_size_factor,
        multi_gpu=multi_gpu,
        use_dist=use_dist,
        min_points_in_gt=prep_cfg.min_num_of_points_in_gt,
        random_flip_x=prep_cfg.random_flip_x,
        random_flip_y=prep_cfg.random_flip_y,
        rand_aug_ratio=prep_cfg.random_aug_ratio,
        sample_importance=prep_cfg.sample_importance,
        rand_rotation_eps=prep_cfg.rand_rotation_eps,
        rand_translation_eps=prep_cfg.rand_translation_eps,
        gen_tq_map=model_config.odom_predictor.pred_pyramid_motion,  # !!!
        do_pre_transform=prep_cfg.do_pre_transform,
        cubic_tq_map=prep_cfg.cubic_tq_map,
        downsample_voxel_sizes=list(prep_cfg.downsample_voxel_sizes)
    )

    dataset = dataset_cls(
        info_path=dataset_cfg.kitti_info_path,
        root_path=dataset_cfg.kitti_root_path,
        seq_length=dataset_cfg.seq_length,
        skip=dataset_cfg.skip,
        random_skip=dataset_cfg.random_skip,
        prep_func=prep_func,
        step=dataset_cfg.step,
        num_point_features=num_point_features,
        split=split, 
    )

    return dataset
