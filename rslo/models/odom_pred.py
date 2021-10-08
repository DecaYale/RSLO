import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import rslo.models.custom_resnet_spc as resnet

from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args
from rslo.layers.SparseConv import SPC_BN2d, SPC_ReLU, SPC_LeakyReLU, SparseConv, SPC_IN2d, SPC_SyncBN2d
from rslo.layers.MaskConv import MaskConv, MaskConvTranspose2d,MaskMaxPool2d
from rslo.utils.pose_utils import rotate_vec_by_q
# from rslo.models.odom_pred import register_odom_pred
# from rslo.data.kitti_dataset import from_pointwise_local_transformation_tch
from rslo.data.dataset import from_pointwise_local_transformation_tch
from rslo.utils.geometric import inverse_warp
import apex 
# from rslo.models.odom_pred_encdec import OdomPredEncDecBase
from rslo.models.odom_pred_base import OdomPredEncDecBase
from rslo.layers.svd import SVDHead 
import kornia
from rslo.utils.geometric import gen_voxel_3d_coords 
from rslo.layers.confidence import ConfidenceModule
import apex.amp as amp



REGISTERED_ODOM_PRED_CLASSES = {}

def register_odom_pred(cls, name=None):
    global REGISTERED_ODOM_PRED_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_ODOM_PRED_CLASSES, f"exist class: {REGISTERED_ODOM_PRED_CLASSES}"
    REGISTERED_ODOM_PRED_CLASSES[name] = cls
    return cls


def get_odom_class(name):
    global REGISTERED_ODOM_PRED_CLASSES
    assert name in REGISTERED_ODOM_PRED_CLASSES, f"available class: {REGISTERED_ODOM_PRED_CLASSES}"
    return REGISTERED_ODOM_PRED_CLASSES[name]

class UNOdomPredEncDecSVDTempMaskBase(OdomPredEncDecBase):

    def __init__(self,
                 use_svd=True,
                 *args,
                 **kwargs):
        super(UNOdomPredEncDecSVDTempMaskBase, self).__init__(*args, **kwargs)
        self.use_svd = use_svd 
        num_upsample_filters= kwargs.get('num_upsample_filters')
        conf_type = kwargs.get('conf_type')

        pyramid_tconf_blocks=[]
        pyramid_qconf_blocks=[]
        pyramid_motion_blocks=[]
        for i, _ in enumerate(num_upsample_filters):
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
                        64,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    self.BatchNorm2d(64),
                    self.ReLU(),
                    nn.Conv2d(
                        64,
                        7,
                        1,
                        stride=1)
                )
                pyramid_motion_blocks.append(py_motion_block)

                pyramid_tconf_blocks.append(
                    ConfidenceModule(
                        nn.Sequential(
                            nn.Conv2d(num_upsample_filters[i],
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
                )
                pyramid_qconf_blocks.append(
                    ConfidenceModule(
                        nn.Sequential(
                            nn.Conv2d(num_upsample_filters[i],
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
                )
                
        self.pyramid_motion_blocks = nn.ModuleList(pyramid_motion_blocks)

        self.q_map_conf = ConfidenceModule(
            nn.Sequential(
            nn.Conv2d(num_upsample_filters[-1], 64, kernel_size=3, padding=1),
            self.BatchNorm2d(64),
            self.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            self.BatchNorm2d(32),
            self.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        ), 
        conf_type = conf_type
        )

        self.t_map_conf = ConfidenceModule(
            nn.Sequential(
            nn.Conv2d(num_upsample_filters[-1], 64, kernel_size=3, padding=1),
            self.BatchNorm2d(64),
            self.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            self.BatchNorm2d(32),
            self.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        ),
        conf_type= conf_type
        )

        self.pyramid_tconf_blocks = nn.ModuleList(pyramid_tconf_blocks)
        self.pyramid_qconf_blocks = nn.ModuleList(pyramid_qconf_blocks)

        self.hier_weight_gen=nn.AvgPool2d(3,2,padding=1)


    @amp.float_function
    def forward(self, xs, tq_map_gt=None,local_spatial_features=None, **kwargs):
        '''xs: a list of inputs'''
        if not isinstance(xs, list):
            xs = [xs]

        ups = []
        stage_outputs = []

        seq_len = len(xs)
        if self._cycle_constraint: #and self.training :
            xs = self.create_cycle_constraint_data(xs)

        input_channel = xs[0].shape[1]
        input_mask_bool = (
            torch.sum(xs[0], dim=1, keepdim=True) != 0).detach_()
        # (torch.sum(x, dim=1, keepdim=True)
        input_mask = input_mask_bool.to(dtype=xs[0].dtype)#float()

        if self._dropout_input:
            xs[0], drop_mask = self.input_drop(xs[0])
            xs[1], _ = self.input_drop(xs[1], drop_mask)


        # group and concatenate the xs
        # x = torch.cat(xs, dim=1)
        # if self._use_correlation:
        #     mask1 = (torch.sum(xs[0],  dim=1, keepdim=True) != 0).to(dtype=xs[0].dtype)#float()
        #     mask2 = (torch.sum(xs[1],  dim=1, keepdim=True) != 0).to(dtype=xs[0].dtype)#.float()  # .detach_()
        #     corr = self.correlation(*xs, mask1, mask2)

        x = torch.cat(xs, dim=1)
        if self._use_correlation:
            x = torch.cat([x,corr], dim=1)
            
        feats = []
        down_masks = []
        if self._use_dynamic_mask:
            x = (x, dynamic_mask*input_mask)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)

            if self._use_sparse_conv or self._use_mask_conv:
                ups.append(self.skip_blocks[i](x[0]))
                down_masks.append(x[1])

            else:
                ups.append(self.skip_blocks[i](x))
                # down_masks.append(None)
        # if len(self.blocks) > 0:
        if (self._use_sparse_conv or self._use_mask_conv):
            # feats.append(x[0])
            x = x[0]

        # x = torch.cat(feats, dim=1)
        
        x_middle = x
        # generate pyramid masks
        if self.pred_pyramid_motion:
            py_masks = []
            p_mask = input_mask#*dynamic_mask
            for i in range(len(self.deblocks)-1):
                p_mask = self.mask_gen_pools[-(i+1)](p_mask)
                py_masks.append(p_mask)
            py_masks.reverse()

        py_preds = []
        for i in range(len(self.deblocks)):
            x = torch.cat([x, ups[-(i+1)]], dim=1)
            x = self.deblocks[i](x)
            if self.pred_pyramid_motion and i < len(self.deblocks)-1:
                py_pred = self.pyramid_motion_blocks[i](x)
                py_preds.append(
                    [py_pred*(py_masks[i]>0).to(dtype=py_pred.dtype), py_masks[i] ] )

        x_tail = x

        tq_map = self.tq_map_conv(x)

        q_map = tq_map[:, 3:] / \
            torch.norm(tq_map[:, 3:], dim=1, keepdim=True)
        t_map = tq_map[:, :3]
        tq_map = torch.cat([t_map, q_map], dim=1)

        odoms=[]
        #TODO: fix me
        tq_map_g=tq_map
        t_conf = torch.ones_like(tq_map[:,:1])
        r_conf = torch.ones_like(tq_map[:,:1])
        if self.dense_predict:
            t_conf = self.t_map_conf(x_tail, extra_mask=input_mask)
            r_conf = self.q_map_conf(x_tail, extra_mask=input_mask)

            tq_map_g = from_pointwise_local_transformation_tch(
                    tq_map, self.point_cloud_range)
            

            tq_maps = [] + [tq_map]
            selected_masks = [] + [input_mask_bool]
            t_confs =  [t_conf]
            r_confs =  [r_conf]
            odoms += self.aggregate_tq(tq_maps, selected_masks=selected_masks,t_confs=t_confs, r_confs=r_confs)

            # temp_t_conf=self.t_map_conf(x_tail.detach(), extra_mask=input_mask, temperature=10) 
            # temp_r_conf = self.q_map_conf(x_tail.detach(), extra_mask=input_mask, temperature=10)
            temp_t_conf=self.t_map_conf(x_tail.detach(), extra_mask=input_mask, temperature=20) 
            temp_r_conf = self.q_map_conf(x_tail.detach(), extra_mask=input_mask, temperature=20)
            temp_tq_conf=torch.cat([temp_t_conf,temp_r_conf],dim=1 ).detach()

            # pyramid_motion=py_preds+[[tq_map*input_mask, input_mask*dynamic_mask*temp_tq_conf ] ]
            pyramid_motion=py_preds+[[tq_map*input_mask, input_mask*temp_tq_conf ] ]
            for p in range(2, len(pyramid_motion)+1):
                pyramid_motion[-p][1] = pyramid_motion[-p][1]*self.hier_weight_gen( pyramid_motion[-(p-1)][1] )
        else:
            py_preds = []
            pyramid_motion=[]
            x = x_middle
            x = self.pool(x)
            x = x.view(x.size(0), -1)

            x = self.fc1(x)
            x = F.relu(x)
            x = self.odom_dropout(x)
            x = self.fc2(x)
            odoms+=[x]

        translations = []
        rotations = []
        for x in odoms: 
            translation = x[:, :3]
            rotation = x[:, 3:]

            if self.odom_format == 'r(x+t)':
                translation = rotate_vec_by_q(translation, rotation)

            if rotation.shape[-1] ==4: #quaternion
                rotation = rotation/(torch.norm(rotation, dim=1, keepdim=True) + 1e-12)

            if not self.training and rotation.shape[-1]==9: 
                rotation_ = kornia.rotation_matrix_to_quaternion(rotation.reshape(-1,3,3) )
                #formatting
                rotation= torch.zeros_like(rotation_)
                rotation[...,0] = rotation_[...,-1]
                rotation[...,1:] = rotation_[...,:-1]

            translations.append(translation)
            rotations.append(rotation)

        ret_dict = {
            "translation_preds": translations,
            "rotation_preds": rotations,
            # ups
            "tq_map_g":tq_map_g*input_mask,
            "pyramid_motion": pyramid_motion, # py_preds+[(tq_map*input_mask, input_mask*dynamic_mask)],
            # "pyramid_motion": [],
            "transformed_inputs": [torch.mean(
                transformed_x1.detach(), dim=1, keepdim=True), torch.mean(
                transformed_x2.detach(), dim=1, keepdim=True)] if self._use_dynamic_mask else None,
            "t_conf": t_conf,  # .detach(),
            "r_conf": r_conf,  # .detach(),
        }
        return ret_dict

    def aggregate_tq(self, tq_maps, selected_masks, t_confs, r_confs):
        assert len(tq_maps) == len(selected_masks) == len(t_confs) == len(r_confs)
        odoms = []
        for tq_map, selected_mask, t_conf, r_conf in zip(tq_maps, selected_masks, t_confs, r_confs):
            if self.use_svd:
                tq_local_map= tq_map.permute(0,2,3,1)
                t_conf_svd = t_conf.permute(0,2,3,1)

                selected_mask = selected_mask.permute(0,2,3,1).expand_as(tq_local_map[:,:,:,:3])
                B = tq_local_map.shape[0]
                xyzv = gen_voxel_3d_coords(tq_local_map,  self.point_cloud_range, return_seq=False)
                tqs=[]
                for b in range(B):
                    scene_flow_seq = tq_local_map[b,:,:,:3][selected_mask[b]].reshape([1,-1,3])
                    # print(tq_local_map.mean())

                    xyzv_b = xyzv[b][selected_mask[b]].reshape(1, -1,3) #BxNx3 # 1xNx3

                    if(xyzv_b.shape[1]<7):
                        print("Warning: the matched points are not enough for svd")

                    svd = SVDHead()
                    R,t = svd(
                        xyzv_b.permute(0,2,1), (xyzv_b-scene_flow_seq).permute(0,2,1), 
                        weight=t_conf_svd[b][selected_mask[b,...,-1:]].reshape(1,-1) 
                        # weight=None
                        )  
                   
                    q=R.reshape([1,9])
                    tq = torch.cat([t,q], dim=1)
                    tqs.append(tq)
                x = svd_odom =torch.cat(tqs, dim=0)#.detach()  #Bx7
            else:
                tq_map_g = from_pointwise_local_transformation_tch(
                        tq_map, self.point_cloud_range)
                t_map_g = tq_map_g[:, :3]
                q_map_g = tq_map_g[:, 3:]
                reduced_t_map_g = torch.sum(t_map_g*t_conf, dim=(2, 3))\
                    / (torch.sum(t_conf, dim=(2, 3))+1e-12)
                reduced_q_map_g = torch.sum(q_map_g*r_conf, dim=(2, 3))\
                    / (torch.sum(r_conf, dim=(2, 3))+1e-12)

                x = torch.cat([reduced_t_map_g, reduced_q_map_g], dim=-1) 

            odoms.append(x)
            
        return odoms

def conv1x1(in_planes, out_planes, stride=1, Conv2d=None, groups=1):
    """1x1 convolution"""
    if Conv2d is None:
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)
    else:
        return Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)


@register_odom_pred
class UNRResNetOdomPredEncDecSVDTempMask(UNOdomPredEncDecSVDTempMaskBase):
    def __init__(self, *args, **kw):
        self.inplanes = -1
        super(UNRResNetOdomPredEncDecSVDTempMask, self).__init__(*args, **kw)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
            # if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d, apex.parallel.SyncBatchNorm) ):
            # elif isinstance(m, (nn.modules.batchnorm._BatchNorm) ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if zero_init_residual:
        for m in self.modules():
            if isinstance(m, resnet.Bottleneck) and hasattr(m.bn3, 'weight') and isinstance(m.bn3, torch.Tensor):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, resnet.BasicBlock) and hasattr(m.bn2, 'weight') and isinstance(m.bn2, torch.Tensor):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1, first_groups=1, use_norm=True):
        self.inplanes = inplanes
        block = resnet.BasicBlock
        downsample = None

        # attention !!

        if self._use_mask_conv:
            conv2d = MaskConv
        elif self._use_sparse_conv:
            conv2d = SparseConv
        else:
            conv2d = nn.Conv2d
        
        if use_norm:#self._enc_use_norm:
            BatchNorm2d = self.BatchNorm2d
        else:
            BatchNorm2d = Empty
            print(f"Warning: BNs are disabled in the encoder of {self}!!!")
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes *
                        block.expansion, stride, Conv2d=conv2d, groups=first_groups),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride,
                            downsample, BN=BatchNorm2d, Conv2d=conv2d, groups=first_groups, use_se=False, use_sa=False))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            if i<num_blocks-1:
                layers.append(block(self.inplanes, planes,
                                    BN=BatchNorm2d, Conv2d=conv2d))
            else:
                layers.append(block(self.inplanes, planes,
                                    BN=BatchNorm2d, Conv2d=conv2d,  use_se=self.use_se, use_sa=self.use_sa))

        return nn.Sequential(*layers), self.inplanes
