import time

import numpy as np
import spconv
import torch
from torch import nn
from torch.nn import functional as F

from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from rslo.utils import torch_timer
from rslo.layers.normalization import SparseInstanceNorm1d
# from rslo.layers.middle_block import DenseMiddleBlock
import copy
import kornia
REGISTERED_MIDDLE_CLASSES = {}


def register_middle(cls, name=None):
    global REGISTERED_MIDDLE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MIDDLE_CLASSES, f"exist class: {REGISTERED_MIDDLE_CLASSES}"
    REGISTERED_MIDDLE_CLASSES[name] = cls
    return cls


def get_middle_class(name):
    global REGISTERED_MIDDLE_CLASSES
    assert name in REGISTERED_MIDDLE_CLASSES, f"available class: {REGISTERED_MIDDLE_CLASSES}"
    return REGISTERED_MIDDLE_CLASSES[name]



@register_middle
class SpMiddleFHDWithCov2_3(nn.Module):
    def __init__(self,
                 output_shape,
                #  use_norm=True,
                 use_GN=False,
                 sync_bn=False, #deprecated
                 bn_type='None',
                 use_leakyReLU=False, #deprecated
                 relu_type='ReLU',
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHDWithConf'):
        super().__init__()

        assert bn_type in ['None','BN', 'SyncBN','SemiGlobalSyncBN','MaskSyncBN'] 
        assert relu_type in ['','ReLU', 'LeakyReLU', "PReLU"]
        self.name = name

        if bn_type !='None': #use_norm:
            if bn_type=='SyncBN' or sync_bn:
                import apex
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(apex.parallel.SyncBatchNorm)
                BatchNorm1d = change_default_args(
                    eps=1e-3, momentum=0.01)(apex.parallel.SyncBatchNorm)
                print("Using SyncBN in middle.")
            elif bn_type =='SemiGlobalSyncBN':
                from rslo.layers.normalization import SemiGlobalSyncBatchNorm
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(SemiGlobalSyncBatchNorm)
                BatchNorm1d = change_default_args(
                    eps=1e-3, momentum=0.01)(SemiGlobalSyncBatchNorm)
                print("Using SemiGlobalSyncBN in middle.")
          

            elif bn_type=='BN':
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
                BatchNorm1d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
                print("Using BN in middle.")
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(
                bias=True, )(spconv.SparseConv3d)
            SubMConv3d = change_default_args(
                bias=True, )(spconv.SubMConv3d)
            ConvTranspose3d = change_default_args(bias=True)(
                 spconv.SparseInverseConv3d)
            print("Set conv bias=True in middle !!? ")
        else:
            print(f"Warning: the batch normalization is turned off in {name}")
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(
                bias=True, )(spconv.SparseConv3d)
            SubMConv3d = change_default_args(
                bias=True, )(spconv.SubMConv3d)
            ConvTranspose3d = change_default_args(bias=True)(
                 spconv.SparseInverseConv3d)
        
        if use_leakyReLU or relu_type=='LeakyReLU':
            self.relu = nn.LeakyReLU
            print("Use leakyReLU in middle")
        elif relu_type=='ReLU':
            self.relu = nn.ReLU
        elif relu_type=='PReLU':
            self.relu = nn.PReLU
            print("Use PReLU in middle")
        else:
            raise ValueError


        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]  # ?
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # input: # [1600, 1200, 41]
        # print(num_input_features, '!!!') #4

        self.middle_conv = spconv.SparseSequential(

            SubMConv3d(num_input_features, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            self.relu(),
            SubMConv3d(16, 16, 3, indice_key="subm0"),
            BatchNorm1d(16),
            self.relu(),
            SpConv3d(16, 32, 3, 2,
                     padding=1,indice_key="conv3d2"),  # [1600, 1200, 41] -> [800, 600, 21]
            # SpConv3d(16, 32, 3, (2,1,1),
            #          padding=1,indice_key="conv3d2"),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(32),
            self.relu(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            self.relu(),
            SubMConv3d(32, 32, 3, indice_key="subm1"),
            BatchNorm1d(32),
            self.relu(),
            SpConv3d(32, 64, 3, 2,
                     padding=1,indice_key="conv3d3"),  # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(64),
            self.relu(),
           
        ) 
        self.middle_conv_tail = spconv.SparseSequential(
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            self.relu(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            self.relu(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            self.relu(),
            SpConv3d(64, 64, 3, 2,
                     padding=[0, 1, 1],indice_key="conv3d4"),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            self.relu(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            self.relu(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            self.relu(),
            SubMConv3d(64, 64, 3, indice_key="subm3"),
            BatchNorm1d(64),
            self.relu(),
            SpConv3d(64, 64, (3, 1, 1),
                     (2, 1, 1),indice_key="conv3d5"),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            self.relu(),
        )
        self.middle_cov_deconv = spconv.SparseSequential(
            # ConvTranspose3d(64, 64, 3, indice_key="conv3d4"),
            # SubMConv3d(64, 64, 3, indice_key="dsubm4"),
            # BatchNorm1d(64),
            # self.relu(),
            ConvTranspose3d(64, 32, 3, 
                     indice_key="conv3d3"),
            # BatchNorm1d(32),
            nn.BatchNorm1d(32),
            self.relu(),
            SubMConv3d(32, 32, 3, indice_key="dsubm3"),
            # BatchNorm1d(32),
            nn.BatchNorm1d(32),
            self.relu(),
            ConvTranspose3d(32, 16, 3, 
                     indice_key="conv3d2"),
            # BatchNorm1d(16),
            nn.BatchNorm1d(16),
            self.relu(),
            SubMConv3d(16, 16, 3, indice_key="dsubm2"),
            # BatchNorm1d(16),
            nn.BatchNorm1d(16),
            self.relu(),
            SubMConv3d(16, 16, 3, indice_key="dsubm2"),
            # BatchNorm1d(8),
            nn.BatchNorm1d(16),
            self.relu(),

            # ConvTranspose3d(16, 16, 3, 
            #          indice_key="conv3d1"),
            # BatchNorm1d(16),
            # # nn.BatchNorm1d(16),
            # self.relu(),
            # SubMConv3d(16, 8, 3, indice_key="dsubm1"),
            # BatchNorm1d(8),
            # # nn.BatchNorm1d(8),
            # self.relu(),
            SubMConv3d(16, 7, 3, indice_key="dsubm1"),
            # nn.Sigmoid()
            # nn.ELU()
        )
        self.max_batch_size = 6
        # self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()
        # self.init_weight()


    def forward(self, voxel_features, coors, batch_size):
                
        assert batch_size==1, "Only support batch_size=1 for now"
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret0 = self.middle_conv(ret)
        
        ret = self.middle_conv_tail(ret0)
        # ret0=copy.deepcopy(ret0)
        ret0.features = ret0.features#.detach()
        cov_pred=self.middle_cov_deconv(ret0)

        
        # cov_pred = eigvec@eigval@eigvec.transpose(-1,-2)
        cov_pred.features[:,:3]= F.elu(cov_pred.features[:,:3])+1+1e-6

        # print("spconv forward time", time.time() - t)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret, cov_pred.features

