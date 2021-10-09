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

"""A function to build localization and classification losses from config."""

from rslo.core import losses
# from rslo.core.ghm_loss import GHMCLoss, GHMRLoss
from rslo.protos import losses_pb2


def build(loss_config):
    """Build losses based on the config.

    Builds classification, localization losses and optionally a hard example miner
    based on the config.

    Args:
      loss_config: A losses_pb2.Loss object.

    """
    

    rotation_loss_func = _build_rotation_loss(
        loss_config.rotation_loss)
    translation_loss_func = _build_translation_loss(
        loss_config.translation_loss)

    if loss_config.pyramid_rotation_loss.loss_type != '':
        pyramid_rotation_loss = _build_rotation_loss(
            loss_config.pyramid_rotation_loss)
    else:
        pyramid_rotation_loss = rotation_loss_func

    if loss_config.pyramid_translation_loss.loss_type != '':
        pyramid_translation_loss = _build_translation_loss(
            loss_config.pyramid_translation_loss)
    else:
        pyramid_translation_loss = translation_loss_func

    consistency_loss = _build_consistency_loss(loss_config.consistency_loss)

    if loss_config.rigid_transform_loss.weight != 0:
        rigid_transform_loss = losses.RigidTransformLoss(
            rotation_loss_func, translation_loss_func, focal_gamma=loss_config.rigid_transform_loss.focal_gamma)
        py_rigid_transform_loss = losses.RigidTransformLoss(
            pyramid_rotation_loss, pyramid_translation_loss, focal_gamma=loss_config.rigid_transform_loss.focal_gamma)

        return (rigid_transform_loss, py_rigid_transform_loss, consistency_loss)

    return (rotation_loss_func, translation_loss_func,
            pyramid_rotation_loss, pyramid_translation_loss, consistency_loss,
            )


def _build_translation_loss(loss_config):
    """Builds a translation loss function based on the loss config.

    Args:
      loss_config: A losses_pb2.TranslationLoss object.

    Returns:
      Loss based on the config.

    Raises:
      ValueError: On invalid loss_config.
    """
    if not isinstance(loss_config, losses_pb2.TranslationLoss):
        raise ValueError(
            'loss_config not of type losses_pb2.TranslationLoss.')

    # loss_config.WhichOneof('localization_loss')
    loss_type = loss_config.loss_type
    loss_weight = loss_config.weight
    focal_gamma = loss_config.focal_gamma
    if loss_type == 'L2':
        return losses.L2Loss(loss_weight)
    elif loss_type == 'AdaptiveWeightedL2':
        if loss_config.balance_scale<=0:
            loss_config.balance_scale=1
        assert loss_config.balance_scale>0
        return losses.AdaptiveWeightedL2Loss(loss_config.init_alpha, learn_alpha=not loss_config.not_learn_alpha, loss_weight=loss_weight, focal_gamma=focal_gamma, balance_scale=loss_config.balance_scale)
    else:
        raise ValueError('Empty loss config.')


def _build_rotation_loss(loss_config):
    """Builds a classification loss based on the loss config.

    Args:
      loss_config: A losses_pb2.RotationLoss object.

    Returns:
      Loss based on the config.

    Raises:
      ValueError: On invalid loss_config.
    """
    if not isinstance(loss_config, losses_pb2.RotaionLoss):
        raise ValueError(
            'loss_config not of type losses_pb2.RotaionLoss.')

    
    loss_type = loss_config.loss_type
    loss_weight = loss_config.weight
    focal_gamma = loss_config.focal_gamma

    if loss_type == 'L2':
        return losses.L2Loss(loss_weight)
    elif loss_type == 'AdaptiveWeightedL2':
        if loss_config.balance_scale<=0:
            loss_config.balance_scale=1
        assert loss_config.balance_scale>0
        return losses.AdaptiveWeightedL2Loss(loss_config.init_alpha, learn_alpha=not loss_config.not_learn_alpha, loss_weight=loss_weight, focal_gamma=focal_gamma, balance_scale=loss_config.balance_scale)
    elif loss_type == 'AdaptiveWeightedL2RMatrixLoss':
        return losses.AdaptiveWeightedL2RMatrixLoss(loss_config.init_alpha, learn_alpha=not loss_config.not_learn_alpha, loss_weight=loss_weight, focal_gamma=focal_gamma)

    raise ValueError('Empty loss config.')


def _build_consistency_loss(loss_config):
    if not isinstance(loss_config, losses_pb2.ConsistencyLoss):
        raise ValueError(
            'loss_config not of type losses_pb2.ConsistencyLoss.')

    loss_type = loss_config.loss_type
    loss_weight = loss_config.weight

    if loss_type=='AdaptiveWeightedL2':
        return losses.AdaptiveWeightedL2Loss(loss_config.init_alpha, learn_alpha=not loss_config.not_learn_alpha, loss_weight=loss_weight, focal_gamma=loss_config.focal_gamma, balance_scale=loss_config.balance_scale )
    elif loss_type=='Aleat5_1ChamferL2NormalWeightedALLSVDLoss':
        assert loss_config.penalize_ratio>0
        assert loss_config.pred_downsample_ratio>0
        assert loss_config.reg_weight >0
        assert loss_config.sph_weight >0
        return losses.Aleat5_1ChamferL2NormalWeightedALLSVDLoss(loss_weight=loss_weight, penalize_ratio=loss_config.penalize_ratio, sample_block_size=loss_config.sample_block_size, norm=loss_config.norm, pred_downsample_ratio=loss_config.pred_downsample_ratio,reg_weight=loss_config.reg_weight, sph_weight=loss_config.sph_weight)
    else: 
        print('Warning: Empty loss config.')
        return None

