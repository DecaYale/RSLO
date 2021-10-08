import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args

REGISTERED_VFE_CLASSES = {}


def register_vfe(cls, name=None):
    global REGISTERED_VFE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_VFE_CLASSES, f"exist class: {REGISTERED_VFE_CLASSES}"
    REGISTERED_VFE_CLASSES[name] = cls
    return cls


def get_vfe_class(name):
    global REGISTERED_VFE_CLASSES
    assert name in REGISTERED_VFE_CLASSES, f"available class: {REGISTERED_VFE_CLASSES}"
    return REGISTERED_VFE_CLASSES[name]


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, name='vfe'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        # [K, 1, units]
        repeated = aggregated.repeat(1, voxel_count, 1)

        concatenated = torch.cat([pointwise, repeated], dim=2)
        # [K, T, 2 * units]
        return concatenated


@register_vfe
class VoxelFeatureExtractor(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) == 2
        num_input_features += 3  # add mean features
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = Linear(num_filters[1], num_filters[1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_dist],
                                 dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(
            num_voxels, voxel_count, axis=0)  # !point-wise mask to remove the padded data in each voxel
        mask = torch.unsqueeze(mask, -1).type_as(features)
        # mask = features.max(dim=2, keepdim=True)[0] != 0
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)
        x *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(x, dim=1)[0]
        return voxelwise


@register_vfe
class VoxelFeatureExtractorV2(nn.Module):
    """VoxelFeatureExtractor with arbitrary number of VFE. deprecated.
    """

    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + num_filters
        filters_pairs = [[num_filters[i], num_filters[i + 1]]
                         for i in range(len(num_filters) - 1)]
        self.vfe_layers = nn.ModuleList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs])
        self.linear = Linear(num_filters[-1], num_filters[-1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[-1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_dist],
                                 dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = self.norm(features.permute(0, 2, 1).contiguous()).permute(
            0, 2, 1).contiguous()
        features = F.relu(features)
        features *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise


@register_vfe
class SimpleVoxel(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(SimpleVoxel, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :self.num_input_features].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()
@register_vfe
class SimpleVoxel_XYZNormalC(nn.Module):
    def __init__(self,
                 num_input_features=8,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(SimpleVoxel_XYZNormalC, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :self.num_input_features].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        points_mean[:, 3:6] = points_mean[:,  3:6] / \
            (torch.norm(points_mean[:,  3:6], dim=-1, keepdim=True)+1e-12)

        return points_mean.contiguous()

@register_vfe
class SimpleVoxel_XYZINormalC(nn.Module):
    def __init__(self,
                 num_input_features=8,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(SimpleVoxel_XYZINormalC, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :self.num_input_features].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        points_mean[:, 4:7] = points_mean[:,  4:7] / \
            (torch.norm(points_mean[:,  4:7], dim=-1, keepdim=True)+1e-12)

        return points_mean.contiguous()
@register_vfe
class SimpleVoxel_XYZINormalNormalGT(nn.Module):
    def __init__(self,
                 num_input_features=8,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='SimpleVoxel_XYZINormalNormalGT'):
        super(SimpleVoxel_XYZINormalNormalGT, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
        assert(self.num_input_features==7)

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        # points_mean = features[:, :, :self.num_input_features].sum(
        # points_mean = features[:, :, :self.num_input_features+3].sum(
        #     dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        points_mean = features[:, :, :].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        points_mean[:, 4:7] = points_mean[:,  4:7] / \
            (torch.norm(points_mean[:,  4:7], dim=-1, keepdim=True)+1e-12)
        
        # print(points_mean.shape, "!!!", flush=True)
        # '''
        points_mean[:, self.num_input_features:]= points_mean[:, self.num_input_features:]/ \
            (torch.norm(points_mean[:, self.num_input_features:], dim=-1, keepdim=True)+1e-12)
        # '''
        return points_mean.contiguous().detach()#,normal_gt.contiguous()
@register_vfe
class SimpleVoxel_XYZINormalC_Normalize(nn.Module):
    def __init__(self,
                 num_input_features=8,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(SimpleVoxel_XYZINormalC_Normalize, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
        self.pc_range = pc_range

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :self.num_input_features].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)

        # print(points_mean[0],'!')
        points_mean[:,:3] = points_mean[:,:3]/torch.tensor(self.pc_range[3:], device=points_mean.device)
        
        #TODO:
        points_mean[:,3]=0 #set intensity value to zero
        points_mean[:, 4:7] = points_mean[:,  4:7] / \
            (torch.norm(points_mean[:,  4:7], dim=-1, keepdim=True)+1e-12)
        return points_mean.contiguous()

@register_vfe
class SimpleVoxel_BoundXYZINormalC(nn.Module):
    def __init__(self,
                 num_input_features=8,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='VoxelFeatureExtractor'):
        super(SimpleVoxel_BoundXYZINormalC, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        # points_mean = features[:, :, :self.num_input_features].sum(
        #     dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        points_range = torch.norm(features[:, :, :3], dim=2, keepdim=True) #NxVx1
        min_ranges, min_inds = torch.min(points_range, dim=1, keepdim=True) #Nx1x1
        new_features_xyzi = torch.gather(features[:,:,:4], dim=1,index=min_inds.repeat(1,1,4) ).squeeze(1)  #Nx1x4
        new_features_nc = features[:, :, 4:].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        

        new_features_nc[:, :3] = new_features_nc[:,  :3] / \
            (torch.norm(new_features_nc[:,  :3], dim=-1, keepdim=True)+1e-12)
        
        new_featrue = torch.cat([new_features_xyzi, new_features_nc], dim=-1)

        return new_featrue.contiguous()


@register_vfe
class SimpleVoxelRadius(nn.Module):
    """Simple voxel encoder. only keep r, z and reflection feature.
    """

    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(32, 128),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='SimpleVoxelRadius'):

        super(SimpleVoxelRadius, self).__init__()

        self.num_input_features = num_input_features
        self.name = name

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :self.num_input_features].sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        feature = torch.norm(points_mean[:, :2], p=2, dim=1, keepdim=True)
        # z is important for z position regression, but x, y is not.
        res = torch.cat([feature, points_mean[:, 2:self.num_input_features]],
                        dim=1)
        return res
