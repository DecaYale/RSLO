import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import rslo.models.custom_resnet_spc as resnet

import rslo.torchplus as torchplus
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args
from rslo.layers.SparseConv import SPC_BN2d, SPC_ReLU, SPC_LeakyReLU, SparseConv, SPC_IN2d, SPC_SyncBN2d, SPC_SemiGlobalSyncBN2d,SPC_MaskSyncBN2d
from rslo.layers.MaskConv import MaskConv, MaskConvTranspose2d, MaskMaxPool2d
from rslo.utils.pose_utils import rotate_vec_by_q
# from rslo.data.kitti_dataset import from_pointwise_local_transformation_tch
from rslo.data.dataset import from_pointwise_local_transformation_tch
from rslo.layers.normalization import SpatialGroupedInstanceNorm2d
from rslo.layers.common import ParameterLayer, Dropout2dGivenMask
# from rslo.layers.correlation import Correlation, MaskCorrelation
from rslo.layers.confidence import ConfidenceModule
import apex
import kornia

class OdomPredEncDecBase(nn.Module):

    def __init__(self,
                 point_cloud_range=None,
                #  use_norm=True,
                 enc_use_norm=True,
                 seq_len=1,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 encode_background_as_zeros=True,
                 use_groupnorm=False,#deprecated
                 bn_type='BN',
                 num_groups=32,
                 dropout=0.2,
                 pooling_type='avg_pool',
                 pooling_size=1,
                 cycle_constraint=False,
                 conv_type="official",
                 odom_format='rx+t',
                 pred_pyramid_motion=False,
                 use_deep_supervision=False,
                 use_loss_mask=True,
                 use_dynamic_mask=False,
                 dense_predict=False,
                 use_correlation=False,
                 conf_type='linear',
                 use_SPGN=False,
                 sync_bn=False,
                 use_leakyReLU=False,
                 dropout_input=False,
                 first_conv_groups=1,
                 use_se=False,
                 use_sa=False,
                 use_svd=False,
                 cubic_pred_height=0,
                 name='odomPred',
                 **kwargs):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(OdomPredEncDecBase, self).__init__()
        assert conv_type in ["official", "sparse_conv", "mask_conv"]
        assert odom_format in ['rx+t', 'r(x+t)']
        assert pooling_type in ['avg_pool', 'max_pool']
        assert bn_type in ['None','BN', 'IN', 'SyncBN','SemiGlobalSyncBN', 'MaskSyncBN'] 
        assert conf_type in ['linear', 'softmax']
        self.conf_type = conf_type
        self._cubic_pred_height = cubic_pred_height
        self.point_cloud_range = point_cloud_range
        self.odom_format = odom_format
        self._use_sparse_conv = False
        self._use_mask_conv = False
        self.dense_predict = dense_predict
        self._use_loss_mask = use_loss_mask
        self._use_dynamic_mask = use_dynamic_mask
        self._dropout_input = dropout_input
        self._first_conv_groups = first_conv_groups
        self.use_se = use_se
        self.use_sa = use_sa
        self.use_svd = use_svd
        self._use_correlation = use_correlation
        self._enc_use_norm = enc_use_norm

        if conv_type == "sparse_conv":
            self._use_sparse_conv = True
        elif conv_type == "mask_conv":
            self._use_mask_conv = True

        if self._use_mask_conv:
            conv2d = MaskConv
        elif self._use_sparse_conv:
            conv2d = SparseConv
        else:
            conv2d = nn.Conv2d

        if bn_type=="None": #not use_norm:
            print(f"Warning: the batch normalization is turned off in {name}")
        if self._use_sparse_conv:
            print(f"Warning: the sparse convolution is used in {name}")

        self._cycle_constraint = cycle_constraint
        self._num_input_features = num_input_features
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        self.pred_pyramid_motion = use_deep_supervision  # attention

        if self._use_sparse_conv or self._use_mask_conv:
            if use_groupnorm or bn_type =="IN":
                print("Using instance normalization in odom_pred!!!", flush=True)
                self.BatchNorm2d = SPC_IN2d
            elif "BN" in bn_type:
                # if sync_bn:
                if bn_type=="SyncBN" or sync_bn:
                    # print("Using syncBN in odom_pred!!!", flush=True)
                    self.BatchNorm2d = SPC_SyncBN2d
                elif bn_type=="SemiGlobalSyncBN":
                    # print("Using SemiGlobalSyncBN in odom_pred!!!", flush=True)
                    self.BatchNorm2d = SPC_SemiGlobalSyncBN2d 
                elif bn_type =='MaskSyncBN':
                    # print("Using MaskSyncBN in middle.", flush=True)
                    self.BatchNorm2d = SPC_MaskSyncBN2d
                elif bn_type=="BN":
                    # print("Using BN in odom_pred!!!", flush=True)
                    self.BatchNorm2d = SPC_BN2d
                
            # print("negative_slope=1e-3...")
            self.ReLU = SPC_ReLU if not use_leakyReLU else change_default_args(negative_slope=1e-3)(SPC_LeakyReLU)
            self.ConvTranspose2d = MaskConvTranspose2d
        else:
            raise NotImplementedError 

        if bn_type!='None':#use_norm:

            self.BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(self.BatchNorm2d)
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            conv2d = change_default_args(bias=True)(conv2d)
            self.ConvTranspose2d = change_default_args(bias=True)(
                self.ConvTranspose2d)
            # print("Conv layer sets bias=True, momentum=0.01")
        else:
            self.BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            self.ConvTranspose2d = change_default_args(bias=True)(
                self.ConvTranspose2d)
            conv2d = change_default_args(bias=True)(conv2d)

        
        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []
        skip_blocks = []
        pyramid_motion_blocks = []
        for i, layer_num in enumerate(layer_nums):
            if i == 0:
                groups_ = first_conv_groups
                use_norm_ = self._enc_use_norm
            else:
                groups_ = 1
                use_norm_ = self._enc_use_norm  # True

            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i], first_groups=groups_, use_norm=use_norm_)
            blocks.append(block)

            if i - self._upsample_start_idx >= 0:
                skip_blocks.append(
                    nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            num_out_filters,
                            kernel_size=3,
                            stride=1, padding=1),
                        self.BatchNorm2d(
                            num_out_filters),
                        self.ReLU()
                    ))

        for i, _ in enumerate(num_upsample_filters):
            if i == 0:
                num_out_filters = num_filters[-1]*2

            else:
                num_out_filters = num_upsample_filters[i -
                                                       1] + num_filters[-(i+1)]

            deblock = nn.Sequential(
                nn.Upsample(scale_factor=upsample_strides[i]),
                nn.Conv2d(
                    num_out_filters,
                    num_upsample_filters[i],
                    kernel_size=3,
                    stride=1, padding=1),
                self.BatchNorm2d(
                    num_upsample_filters[i]),
                self.ReLU(),
            )
            deblocks.append(deblock)

            if self.pred_pyramid_motion:
                py_motion_block = nn.Sequential(
                    nn.Conv2d(
                        num_upsample_filters[i],
                        num_upsample_filters[i]//2,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    self.BatchNorm2d(num_upsample_filters[i]//2),
                    self.ReLU(),
                    nn.Conv2d(
                        num_upsample_filters[i]//2,
                        7,
                        1,
                        stride=1)
                )
                pyramid_motion_blocks.append(py_motion_block)
                # self.mask_gen_pool = nn.MaxPool2d(
                #     kernel_size=3, stride=2, padding=1)
                self.mask_gen_pools = nn.ModuleList(
                    [nn.MaxPool2d(
                        kernel_size=3, stride=upsample_strides[i], padding=1) for i in range(len(upsample_strides))]
                )

        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        self.skip_blocks = nn.ModuleList(skip_blocks)
        self.pyramid_motion_blocks = nn.ModuleList(pyramid_motion_blocks)

        self.tq_map_conv = nn.Sequential(
            nn.Conv2d(num_upsample_filters[-1], 64, kernel_size=3, padding=1),
            self.BatchNorm2d(64),
            self.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            self.BatchNorm2d(32),
            self.ReLU(),
            nn.Conv2d(32, 7, kernel_size=1),

        )

        self.q_map_conf = ConfidenceModule(
            nn.Sequential(
                nn.Conv2d(num_upsample_filters[-1],
                          64, kernel_size=3, padding=1),
                self.BatchNorm2d(64),
                self.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                self.BatchNorm2d(32),
                self.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),
            ),
            conf_type=conf_type
        )

        self.t_map_conf = ConfidenceModule(
            nn.Sequential(
                nn.Conv2d(num_upsample_filters[-1],
                          64, kernel_size=3, padding=1),
                self.BatchNorm2d(64),
                self.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                self.BatchNorm2d(32),
                self.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),
            ),
            conf_type=conf_type
        )

        if pooling_type == 'avg_pool':
            self.pool = nn.AdaptiveAvgPool2d((pooling_size, pooling_size))
        else:
            self.pool = nn.AdaptiveMaxPool2d((pooling_size, pooling_size))

        # self.pool = nn.AdaptiveAvgPool2d((avgpool, avgpool))
        self.fc1 = nn.Linear(
            num_filters[-1]*pooling_size*pooling_size*seq_len, 1024)

        assert dropout > 0
        self.odom_dropout = nn.Dropout(p=dropout)
        self.dense_dropout = nn.Dropout2d(p=dropout)
        self.fc2 = nn.Linear(1024, 7)  # [translation,rotation]
        self.softmax = nn.Softmax(dim=-1)
        self._use_SPGN = use_SPGN
        if self._use_SPGN:
            print(f"Warning: using SPGN in {name}")
            self.SPGN = SpatialGroupedInstanceNorm2d(
                [1, 5], num_channels=num_input_features)
        else:
            self.SPGN = Empty()
        if 1:  # self._use_dynamic_mask:
            self.dynamic_sigma = ParameterLayer(
                torch.ones(1)*0.1, requires_grad=True)
        if self._dropout_input:
            self.input_drop = Dropout2dGivenMask(p=0.1, dim=1)

    def create_cycle_constraint_data(self, xs):
        '''xs: a list of x in size of (N,C,H,W)'''
        assert len(xs) >= 2
        seq_len = len(xs)
        batch_size, C, H, W = xs[0].shape
        # input_shape=xs[0].shape
        x1 = []
        x2 = []
        for i in range(0, seq_len):
            for j in range(i+1, seq_len):
                x1.append(xs[i])
                x2.append(xs[j])

        x1 = torch.stack(x1, dim=1).reshape(-1, C, H, W)
        x2 = torch.stack(x2, dim=1).reshape(-1, C, H, W)
        # x1 = torch.stack(x1, dim=1).reshape(-1, *input_shape[1:])
        # x2 = torch.stack(x2, dim=1).reshape(-1, *input_shape[1:])

        # print(x1.shape, x2.shape, '!!!')
        return [x1, x2]

    def unravel_prediction(self, pred, seq_len):
        batch_size = pred.shape[0]//seq_len
        preds = [pred[i*batch_size:(i+1)*batch_size]
                 for i in range(seq_len)]
        preds = torch.cat(preds, dim=1)

        return preds

   


def conv1x1(in_planes, out_planes, stride=1, Conv2d=None, groups=1):
    """1x1 convolution"""
    if Conv2d is None:
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)
    else:
        return Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)


