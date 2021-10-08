# Copyright 2017 yanyan. All Rights Reserved.
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
"""VoxelNet builder.
"""

from rslo.protos import second_pb2
from rslo.builder import losses_builder
from rslo.models.voxel_odom_net import  get_voxelnet_class
import rslo.models.voxel_odom_net
# import rslo.models.voxel_odom_net_self_icp_enc
# import rslo.models.voxel_odom_net_self_icp3_enc
# import rslo.models.voxel_odom_net_self_icp_enc2

def build(model_cfg: second_pb2.VoxelNet, voxel_generator,
          measure_time=False, testing=False):
    """build second pytorch instance.
    """
    if not isinstance(model_cfg, second_pb2.VoxelNet):
        raise ValueError('model_cfg not of type ' 'second_pb2.VoxelNet.')
    vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
    vfe_with_distance = model_cfg.voxel_feature_extractor.with_distance  # ??
    grid_size = voxel_generator.grid_size

    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    pc_range = voxel_generator.point_cloud_range
    print(dense_shape, '!!!', flush=True)  # [1, 40, 1600, 1408, 16] [1, z,y,x, 16]

    num_input_features = model_cfg.num_point_features



    losses = losses_builder.build(model_cfg.loss)
    # encode_rad_error_by_sin = model_cfg.encode_rad_error_by_sin
    # cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses
    # rotation_loss_func, translation_loss_func, pyramid_rotation_loss_func, pyramid_translation_loss_func, consistency_loss = losses
    rigid_transform_loss=None
    py_rigid_transform_loss=None
    rotation_loss_func=None 
    translation_loss_func=None 
    pyramid_rotation_loss_func=None 
    pyramid_translation_loss_func=None 
    consistency_loss=None
    if len(losses) == 3:
        rigid_transform_loss, py_rigid_transform_loss, consistency_loss = losses
        
    else:
        rotation_loss_func, translation_loss_func, pyramid_rotation_loss_func, pyramid_translation_loss_func, consistency_loss = losses

    net = get_voxelnet_class(model_cfg.network_class_name)(
        dense_shape,
        pc_range=pc_range,
        vfe_class_name=model_cfg.voxel_feature_extractor.module_class_name,
        vfe_num_filters=vfe_num_filters,
        middle_class_name=model_cfg.middle_feature_extractor.module_class_name,
        middle_num_input_features=model_cfg.middle_feature_extractor.num_input_features,
        middle_num_filters_d1=list(
            model_cfg.middle_feature_extractor.num_filters_down1),
        middle_num_filters_d2=list(
            model_cfg.middle_feature_extractor.num_filters_down2),
        middle_use_leakyReLU=model_cfg.middle_feature_extractor.use_leakyReLU,
        middle_relu_type=model_cfg.middle_feature_extractor.relu_type,
        odom_class_name=model_cfg.odom_predictor.module_class_name,  # "ResNetOdomPred",
        odom_num_input_features=model_cfg.odom_predictor.num_input_features,  # -1,
        odom_layer_nums=model_cfg.odom_predictor.layer_nums,  # [3, 5, 5],
        # [2, 2, 2],
        odom_layer_strides=model_cfg.odom_predictor.layer_strides,
        # [128, 128, 256],
        odom_num_filters=model_cfg.odom_predictor.num_filters,
        # [1, 2, 4],
        odom_upsample_strides=model_cfg.odom_predictor.upsample_strides,
        # [256, 256, 256],
        odom_num_upsample_filters=model_cfg.odom_predictor.num_upsample_filters,
        odom_pooling_size=model_cfg.odom_predictor.pool_size,
        odom_pooling_type=model_cfg.odom_predictor.pool_type,
        odom_cycle_constraint=model_cfg.odom_predictor.cycle_constraint,
        odom_conv_type=model_cfg.odom_predictor.conv_type,
        odom_format=model_cfg.odom_predictor.odom_format,
        # odom_use_spc=model_cfg.odom_predictor.use_sparse_conv,
        odom_pred_pyramid_motion=model_cfg.odom_predictor.pred_pyramid_motion,
        odom_use_deep_supervision=model_cfg.odom_predictor.use_deep_supervision,
        odom_use_loss_mask=not model_cfg.odom_predictor.not_use_loss_mask,
        odom_use_dynamic_mask=model_cfg.odom_predictor.use_dynamic_mask,
        odom_dense_predict=model_cfg.odom_predictor.dense_predict,
        odom_use_corr=model_cfg.odom_predictor.use_corr,
        odom_dropout=model_cfg.odom_predictor.dropout,
        odom_conf_type=model_cfg.odom_predictor.conf_type,
        odom_use_SPGN=model_cfg.odom_predictor.use_SPGN,
        odom_use_leakyReLU=model_cfg.odom_predictor.use_leakyReLU,
        vfe_use_norm=not model_cfg.voxel_feature_extractor.not_use_norm,  # True,
        # middle_use_norm=not model_cfg.middle_feature_extractor.not_use_norm,  # True,
        # odom_use_norm=not model_cfg.odom_predictor.not_use_norm,  # True,
        middle_bn_type = model_cfg.middle_feature_extractor.bn_type,
        odom_bn_type = model_cfg.odom_predictor.bn_type,
        odom_enc_use_norm = not model_cfg.odom_predictor.not_use_enc_norm,
        odom_dropout_input=model_cfg.odom_predictor.dropout_input,
        odom_first_conv_groups=max(
            1, model_cfg.odom_predictor.first_conv_groups),
        odom_use_se=model_cfg.odom_predictor.odom_use_se,
        odom_use_sa=model_cfg.odom_predictor.odom_use_sa,
        odom_use_svd=model_cfg.odom_predictor.use_svd,
        odom_cubic_pred_height=model_cfg.odom_predictor.cubic_pred_height,
        freeze_bn = model_cfg.freeze_bn,
        freeze_bn_affine=model_cfg.freeze_bn_affine,
        freeze_bn_start_step=model_cfg.freeze_bn_start_step,
        # sync_bn=model_cfg.sync_bn,
        use_GN=model_cfg.use_GN,
        num_input_features=num_input_features,

        encode_background_as_zeros=model_cfg.encode_background_as_zeros,

        with_distance=vfe_with_distance,
        rotation_loss=rotation_loss_func,
        translation_loss=translation_loss_func,
        pyramid_rotation_loss=pyramid_rotation_loss_func,
        pyramid_translation_loss=pyramid_translation_loss_func,
        rigid_transform_loss=rigid_transform_loss,
        pyramid_rigid_transform_loss = py_rigid_transform_loss,
        consistency_loss=consistency_loss,
        measure_time=measure_time,
        voxel_generator=voxel_generator,
        pyloss_exp_w_base=model_cfg.loss.pyloss_exp_w_base,
        testing=testing,
        icp_iter=model_cfg.icp_iter,
    )
    return net