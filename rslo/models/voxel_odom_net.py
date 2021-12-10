import rslo.core.losses as losses
# from rslo.models.voxel_odom_net import register_voxelnet
import kornia
# from rslo.data.kitti_dataset import generate_pointwise_local_transformation_tch
from rslo.data.dataset import generate_pointwise_local_transformation_tch
from rslo.utils.geometric import inverse_warp
from rslo.utils.kitti_evaluation import kittiOdomEval
import time
import collections
import contextlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from rslo.models import middle, voxel_encoder  # pointpillars
from rslo.utils import util
from torchplus import metrics
from rslo.utils import torch_timer
from . import odom_pred
# from . import odom_pred_encdec
# from . import odom_pred_encdec_svd_self
# from . import odom_pred_encdec_svd_self_tempmask
import apex.amp as amp
import pickle

REGISTERED_NETWORK_CLASSES = {}

def register_voxelnet(cls, name=None):
    global REGISTERED_NETWORK_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_NETWORK_CLASSES, f"exist class: {REGISTERED_NETWORK_CLASSES}"
    REGISTERED_NETWORK_CLASSES[name] = cls
    return cls


def get_voxelnet_class(name):
    global REGISTERED_NETWORK_CLASSES
    assert name in REGISTERED_NETWORK_CLASSES, f"available class: {REGISTERED_NETWORK_CLASSES}"
    return REGISTERED_NETWORK_CLASSES[name]


TODO_cnt=0

@register_voxelnet
class UnVoxelOdomNetICP3(nn.Module):
    def __init__(self,
                 output_shape,
                 pc_range=None,
                 #  num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 middle_use_leakyReLU=False,  # deprecated
                 middle_bn_type="BN",
                 middle_relu_type="ReLU",
                 odom_class_name="ResNetOdomPred",
                 odom_num_input_features=-1,
                 odom_layer_nums=[3, 5, 5],
                 odom_layer_strides=[2, 2, 2],
                 odom_num_filters=[128, 128, 256],
                 odom_upsample_strides=[1, 2, 4],
                 odom_num_upsample_filters=[256, 256, 256],
                 odom_pooling_type='avg_pool',
                 odom_pooling_size=1,
                 odom_cycle_constraint=False,
                 odom_conv_type='official',
                 odom_format='rx+t',
                 odom_pred_pyramid_motion=False,
                 odom_use_deep_supervision=False,
                 odom_dense_predict=False,
                 odom_use_loss_mask=True,
                 odom_use_dynamic_mask=False,
                 odom_use_corr=False,
                 odom_dropout=0.2,
                 odom_bn_type='BN',
                 odom_conf_type='linear',
                 odom_use_SPGN=False,
                 odom_use_leakyReLU=False,
                 odom_first_conv_groups=1,
                 odom_use_se=False,
                 odom_use_sa=False,
                 vfe_use_norm=True,
                 odom_enc_use_norm=True,
                 odom_use_svd=False,
                 odom_dropout_input=False,
                 odom_cubic_pred_height=5,
                 freeze_bn=False,
                 freeze_bn_affine=False,
                 freeze_bn_start_step=1e20,
                 sync_bn=False,
                 use_GN=False,
                 encode_background_as_zeros=True,
                 rotation_loss=None,
                 translation_loss=None,
                 pyramid_rotation_loss=None,
                 pyramid_translation_loss=None,
                 consistency_loss=None,
                 measure_time=False,
                 voxel_generator=None,
                 pyloss_exp_w_base=0.5,
                 testing=False,
                 icp_iter=2,
                 name='voxel_odom_net',
                 **kwargs):
        super().__init__()


        self.name = name
        self.testing = testing
        self._encode_background_as_zeros = encode_background_as_zeros
        self._num_input_features = num_input_features
        self.voxel_generator = voxel_generator

        self._rotation_loss = rotation_loss
        self._translation_loss = translation_loss
        self._pyramid_rotation_loss = pyramid_rotation_loss
        self._pyramid_translation_loss = pyramid_translation_loss
        self._consistency_loss = consistency_loss
        self._conf_reg_loss = nn.MSELoss

        assert(pyloss_exp_w_base > 0)
        self._pyloss_exp_w_base = pyloss_exp_w_base if pyloss_exp_w_base > 0 else 0.5
        assert icp_iter>0, "The parameter of icp_iter should be larger than 0."
        self.icp_iter = icp_iter

        self.measure_time = measure_time
        self.voxel_feature_extractor = voxel_encoder.get_vfe_class(vfe_class_name)(
            num_input_features,
            vfe_use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=self.voxel_generator.voxel_size,
            pc_range=self.voxel_generator.point_cloud_range,
        )
        self.middle_feature_extractor = middle.get_middle_class(middle_class_name)(
            output_shape,
            # middle_use_norm,
            bn_type=middle_bn_type,
            use_GN=use_GN,
            sync_bn=sync_bn,
            use_leakyReLU=middle_use_leakyReLU,
            relu_type=middle_relu_type,
            num_input_features=middle_num_input_features,
            num_filters_down1=middle_num_filters_d1,
            num_filters_down2=middle_num_filters_d2)

        odom_pred_input_feature = odom_num_input_features*2  # 128*2

        self.middle_feature_extractor_name = middle_class_name
        self.odom_predictor = odom_pred.get_odom_class(odom_class_name)(  # TODO: fix the hard coding
            # use_norm=odom_use_norm,  # True,
            bn_type=odom_bn_type,
            enc_use_norm=odom_enc_use_norm,
            # use_sparse_conv=odom_use_spc,
            conv_type=odom_conv_type,
            layer_nums=odom_layer_nums,  # (3, 5, 5),
            layer_strides=odom_layer_strides,  # (2, 2, 2),
            num_filters=odom_num_filters,  # (128, 128, 256),
            upsample_strides=odom_upsample_strides,  # (1, 2, 4),
            num_upsample_filters=odom_num_upsample_filters,  # (256, 256, 256),
            num_input_features=odom_pred_input_feature,  # 128
            pooling_type=odom_pooling_type,
            pooling_size=odom_pooling_size,
            encode_background_as_zeros=True,
            use_groupnorm=use_GN,
            num_groups=32,
            dropout=odom_dropout,  # 0.2,
            cycle_constraint=odom_cycle_constraint,
            pred_pyramid_motion=odom_pred_pyramid_motion,
            use_deep_supervision=odom_use_deep_supervision,
            use_loss_mask=odom_use_loss_mask,
            use_dynamic_mask=odom_use_dynamic_mask,
            odom_format=odom_format,
            point_cloud_range=pc_range,
            dense_predict=odom_dense_predict,
            use_correlation=odom_use_corr,
            conf_type=odom_conf_type,
            use_SPGN=odom_use_SPGN,
            use_leakyReLU=odom_use_leakyReLU,
            dropout_input=odom_dropout_input,
            first_conv_groups=odom_first_conv_groups,
            use_se=odom_use_se,
            use_sa=odom_use_sa,
            use_svd=odom_use_svd,
            cubic_pred_height=odom_cubic_pred_height,
            freeze_bn=freeze_bn,
            freeze_bn_affine=freeze_bn_affine,
            sync_bn=sync_bn,
            name='odomPred'
        )

        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        self.freeze_bn_start_step = freeze_bn_start_step

        self.register_buffer("global_step", torch.LongTensor(1).zero_())
        # self.global_step= 0

        self.warm_flag = False

        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}

    def train(self, mode=True):
        """
            Override the default train() to freeze the BN parameters
        """

        super(UnVoxelOdomNetICP3, self).train(mode)
        if self.freeze_bn and self.get_global_step() >= self.freeze_bn_start_step:
            print("New version: Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn_affine:
                print("New version: Freezing Weight/Bias of BatchNorm2D.")
        tmp_flag = True
        if self.freeze_bn and self.get_global_step() >= self.freeze_bn_start_step:
            # for m in self.backbone.modules():
            for m in self.modules():
                if isinstance(m, (nn.modules.batchnorm._BatchNorm)):
                    m.eval()
                    if tmp_flag:
                        tmp_flag = False
                        print(m.weight[:3], m.bias[:3], m.running_mean[:3],
                              m.running_var[:3], '!!', flush=True)

                    if self.freeze_bn_affine:
                        if m.weight is not None:
                            m.weight.requires_grad = False
                        if m.bias is not None:
                            m.bias.requires_grad = False
        return self

    def start_timer(self, *names):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        for name in names:
            self._time_dict[name] = time.time()

    def end_timer(self, name):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()

    @contextlib.contextmanager
    def profiler(self):
        old_measure_time = self.measure_time
        self.measure_time = True
        yield
        self.measure_time = old_measure_time

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def clear_global_step(self):
        self.global_step.zero_()

    def gen_dynamic_mask(self, features1, features2, tq_map):
        warped_feature2, valid_points = inverse_warp(
            features1, features2, tq_map, pc_range=self.odom_predictor.point_cloud_range)

    def gen_tq_maps(self, odometries, spatial_size, pc_range, cubic_tq_map=False):

        if len(spatial_size) == 2:
            spatial_size = [1]+list(spatial_size)
        # np.array(voxel_generator.grid_size)//4
        grid_size = np.array(list(spatial_size[::-1]))
        voxel_size = (pc_range[3:]-pc_range[0:3])/grid_size
        # pc_range = voxel_generator.point_cloud_range
        if cubic_tq_map:
            spatial_size = grid_size
            # generate_pointwise_local_transformation_3d
            tq_map_generator = generate_pointwise_local_transformation_tch
        else:
            spatial_size = grid_size[:2]
            tq_map_generator = generate_pointwise_local_transformation_tch
        origin_loc = (0-pc_range[0])/(pc_range[3]-pc_range[0])*(grid_size[0]
                                                                ), (pc_range[4]-0)/(pc_range[4]-pc_range[1])*(grid_size[1]), (0-pc_range[2])/(pc_range[5]-pc_range[2]) * grid_size[2]

        tq_maps = []
        for tq in odometries:
            tq_maps.append(tq_map_generator(
                tq,
                # spatial_size=grid_size[:2],
                spatial_size=spatial_size,
                origin_loc=origin_loc,  # np.array(origin_loc),
                voxel_size=voxel_size,
                inv_trans_factor=-1
            ))
        tq_maps = torch.stack(tq_maps, dim=0)
        return [tq_maps]

    @amp.float_function
    def loss(self, example, preds_dict):

        T_preds = preds_dict["translation_preds"]
        R_preds = preds_dict["rotation_preds"]

        dtype = T_preds[0].dtype
        if not isinstance(T_preds, (list, tuple)):
            T_preds = [T_preds]
        if not isinstance(R_preds, (list, tuple)):
            R_preds = [R_preds]

        pyramid_preds = preds_dict["pyramid_motion"]


        self.start_timer("Create_loss forward")
        pyramid_loss = torch.zeros([1], dtype=dtype).cuda()

        translation_loss, rotation_loss, pyramid_T_losses, pyramid_R_losses, C_loss = self.create_loss(
            preds_dict,
            example,
            self._translation_loss,
            self._rotation_loss,
            pyramid_rotation_loss=self._pyramid_rotation_loss,
            pyramid_translation_loss=self._pyramid_translation_loss,
            consistency_loss=self._consistency_loss
        )
        pyramid_num = len(pyramid_T_losses)
        for i, (t_loss, r_loss) in enumerate(zip(pyramid_T_losses, pyramid_R_losses)):
            # w = 0.5**(pyramid_num-i)
            w = self._pyloss_exp_w_base**(pyramid_num-i)
            pyramid_loss += w*(t_loss + r_loss)

        # local_num = len(local_T_losses)
        # for i, (t_loss, r_loss) in enumerate(zip(local_T_losses, local_R_losses)):
        #     # local_loss += 0.5**(i+1)* (t_loss + r_loss)
        #     local_loss += 0.2**(local_num-i) * (t_loss + r_loss)
        #     print("!!!!", flush=True)
       
        loss = translation_loss + rotation_loss + pyramid_loss + C_loss
        self.end_timer("create_loss forward")

        # self.end_timer("loss forward")
        res = {
            "loss": loss,
            "translation_loss": translation_loss.detach(),
            "rotation_loss": rotation_loss.detach(),
            "pyramid_loss": pyramid_loss.detach(),
            "C_loss": C_loss.detach(),
            "translation_preds": T_preds[0].detach(),
            "rotation_preds": R_preds[0].detach(),
        }
        return res

    def network_forward(self, voxels, num_points, coors, batch_size, example):
        """this function is used for subclass.
        you can add custom network architecture by subclass VoxelNet class
        and override this function.
        voxels: [voxels_t0, voxels_t1 ...]
        num_points: [num_points_t0, num_points_t1, ...]
        coors: [coors_t0, coors_t1, ...]
        batch_size: [batch_size_t0, batch_size_t1, ...]
        Returns:
            preds_dict: {
                box_preds: ...
                cls_preds: ...
                dir_cls_preds: ...
            }
        """
        assert len(voxels) == len(num_points) == len(
            coors), "The lengths should be same."

        # self.measure_time=True
        self.start_timer("voxel_feature_extractor")
        voxel_features = []
        normal_gts=[]
        if self.voxel_feature_extractor.name in ['SimpleVoxel_XYZINormalNormalGT']:
            for i in range(len(voxels)):
                # fea,normal_gt = self.voxel_feature_extractor(voxels[i], num_points[i],
                #                                 coors[i])
                fea = self.voxel_feature_extractor(voxels[i], num_points[i],
                                                coors[i])
                # voxel_features.append(fea[:,:-3])
                voxel_features.append(fea[:,:7].clone())
                # print(fea.shape,fea[100,:])
                # voxel_features.append(fea[:,:])
                normal_gts.append(fea[:,-3:].clone())
            example['normal_gt']= normal_gts
        else:
            for i in range(len(voxels)):
                voxel_features.append(
                    self.voxel_feature_extractor(voxels[i], num_points[i],
                                                coors[i]))
        self.end_timer("voxel_feature_extractor")
        # print(self._time_total_dict["voxel_feature_extractor"]/self._time_count_dict["voxel_feature_extractor"] )
        

        self.start_timer("middle forward")

        spatial_features = []
        middle_conf_preds=[]
        normal_preds=[]
        if self.middle_feature_extractor_name in ['SpMiddleFHD_Early', 'SpMiddleFHD_2to1', 'SpMiddleFHD_2to1_v2', 'SpMiddleFHD_2to1_v3']:

            spatial_features.append(self.middle_feature_extractor(
                voxel_features[-2:], coors[-2:], batch_size))
        elif self.middle_feature_extractor_name in [ 'SpMiddleFHDWithCov2_3' ]: #"SpMiddleFHDWithConf",
            for i in range(len(voxel_features)):
                ret, conf_pred=self.middle_feature_extractor(
                        voxel_features[i], coors[i], batch_size)
                spatial_features.append(ret)  # ! NxCxHxW
                middle_conf_preds.append(conf_pred)
        elif self.middle_feature_extractor_name in ['SpMiddleFHDWithCov2_2_1']:
            for i in range(len(voxel_features)):
                ret, conf_pred, normal_pred=self.middle_feature_extractor(
                        voxel_features[i], coors[i], batch_size)
                spatial_features.append(ret)  # ! NxCxHxW
                middle_conf_preds.append(conf_pred)
                normal_preds.append(normal_pred)

        else:
            for i in range(len(voxel_features)):
                spatial_features.append(
                    self.middle_feature_extractor(
                        voxel_features[i], coors[i], batch_size))  # ! NxCxHxW

        self.end_timer("middle forward")

        preds_dict = self.odom_predictor(
            spatial_features, tq_map_gt=example['tq_maps'][0])  # .view(-1, *example['tq_maps'][0].shape[2:])) # modified on 21/12/2019

        mask_display = (torch.sum(torch.cat(spatial_features,
                                            dim=1).detach(), dim=1, keepdim=True) != 0).float()
        preds_dict['feature_mask'] = mask_display
        feature_display = [torch.mean(
            spatial_features[i].detach(), dim=1, keepdim=True).detach() for i in range(len(spatial_features))]
        preds_dict['middle_feature'] = [
            (feature_display[i]-torch.min(feature_display[i])) /
            (torch.max(feature_display[i]) -
             torch.min(feature_display[i])+1e-12)
            for i in range(len(feature_display))]
        
        preds_dict['middle_conf_preds'] = middle_conf_preds
        preds_dict['voxel_features'] = voxel_features
        preds_dict['voxel_coords'] = coors
        preds_dict['normal_preds'] = normal_preds
        return preds_dict

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]  # ! [Nx(max_points_per_voxel)x(points_dim), ...]
        # ! (N,), the actual #points per voxel
        num_points = example["num_points"]
        coors = example["coordinates"]  # ! (N,), the coordinate for each voxel
        if len(num_points[0].shape) == 2:  # multi-gpu
            # process each frame in a sequence with different timestamps
            voxels_buf = []
            num_points_buf = []
            coors_buf = []
            for t in range(len(voxels)):
                num_voxel_per_batch = example["num_voxels"][t].cpu().numpy().reshape(
                    -1)
                # import pdb
                # pdb.set_trace()

                voxel_list = []
                num_points_list = []
                coors_list = []

                for i, num_voxel in enumerate(num_voxel_per_batch):
                    voxel_list.append(voxels[t][i, :num_voxel])
                    num_points_list.append(num_points[t][i, :num_voxel])
                    coors_list.append(coors[t][i, :num_voxel])
                # voxels = torch.cat(voxel_list, dim=0)
                # num_points = torch.cat(num_points_list, dim=0)
                # coors = torch.cat(coors_list, dim=0)
                voxels_buf.append(torch.cat(voxel_list, dim=0))
                num_points_buf.append(torch.cat(num_points_list, dim=0))
                coors_buf.append(torch.cat(coors_list, dim=0))
            voxels = voxels_buf
            num_points = num_points_buf
            coors = coors_buf

        batch_size_dev = example["num_voxels"][0].shape[0]

        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        preds_dict = self.network_forward(
            voxels, num_points, coors, batch_size_dev, example=example)
        # need to check size.
        if self.training:
            ret = self.loss(example, preds_dict)

            ret2 = {}
            ret2['middle_feature'] = preds_dict['middle_feature']
            ret2['feature_mask'] = preds_dict['feature_mask']
            ret2['t_conf'] = preds_dict.pop('t_conf', None).detach()
            ret2['r_conf'] = preds_dict.pop('r_conf', None).detach()
            # ret['tq_map'] = preds_dict.pop('tq_map', None)
            ret2['pyramid_motion'] = preds_dict.pop('pyramid_motion', None)
            ret2['dynamic_sigma'] = preds_dict.pop('dynamic_sigma', -1)
            ret2['transformed_inputs'] = preds_dict.pop(
                'transformed_inputs', None)
            ret2['tq_map_g'] = preds_dict.pop('tq_map_g', None).detach()
            ret2['local_motion'] = preds_dict.pop('local_motion', None)
            ret2['down_masks'] = preds_dict.pop('down_masks', None)
            # ret2['middle_conf_preds'] =[p.features for p in preds_dict['middle_conf_preds']]
            ret2['middle_conf_preds'] =[p for p in preds_dict['middle_conf_preds']]

            ret2 = util.dict_recursive_op(
                ret2, op=lambda x: [xx.cpu().detach() for xx in x] if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor) else x)
            ret2 = util.dict_recursive_op(
                ret2, op=lambda x: x.cpu().detach() if isinstance(x, torch.Tensor) else x)
            ret.update(ret2)
            return ret
        else:
            # ret = self.loss(example, preds_dict)
            if not self.testing:
                preds_dict.pop('tq_map_g', None)
                preds_dict.pop('pyramid_motion', None)
                preds_dict.pop('local_motion', None)
                preds_dict.pop('middle_feature', None)
                preds_dict.pop('feature_mask', None)
                preds_dict.pop('t_conf', None)
                preds_dict.pop('r_conf', None)
                preds_dict.pop('transformed_input', None)
                preds_dict.pop('down_masks', None)
                
            if isinstance(preds_dict['translation_preds'], (list, tuple)):
                # preds_dict['translation_preds'] = preds_dict['translation_preds'][0]
                preds_dict['translation_preds'] = preds_dict['translation_preds'][-1]
            if isinstance(preds_dict['rotation_preds'], (list, tuple)):
                # preds_dict['rotation_preds'] = preds_dict['rotation_preds'][0]
                preds_dict['rotation_preds'] = preds_dict['rotation_preds'][-1]

            preds_dict = util.dict_recursive_op(
                preds_dict, op=lambda x: [xx.detach() for xx in x] if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor) else x)
            preds_dict = util.dict_recursive_op(
                preds_dict, op=lambda x: x.detach() if isinstance(x, torch.Tensor) else x)

            ret_pred_dict = {}
            ret_pred_dict['translation_preds'] = preds_dict['translation_preds'].detach()
            ret_pred_dict['rotation_preds'] = preds_dict['rotation_preds'].detach()
            if self.testing:
                ret_pred_dict['middle_conf_preds'] =[p for p in preds_dict['middle_conf_preds']]
                ret_pred_dict['voxel_features'] = preds_dict['voxel_features'] 
                ret_pred_dict['normal_preds'] = preds_dict['normal_preds'] 
                
            
                ret_pred_dict['tq_map_g']=preds_dict.get('tq_map_g',None).detach()
                ret_pred_dict['pyramid_motion']=preds_dict.get('pyramid_motion', None)
                ret_pred_dict['t_conf'] = preds_dict.get('t_conf', None)
                ret_pred_dict['r_conf'] = preds_dict.get('r_conf', None)
                ret_pred_dict['normal_gt'] = example.get('normal_gt', None)
        return ret_pred_dict


    def clear_metrics(self):
        pass

    @amp.float_function
    def create_loss(self,
                    preds_dict,
                    example,
                    translation_loss,
                    rotation_loss,
                    pyramid_translation_loss=None,
                    pyramid_rotation_loss=None,
                    pyramid_preds=None,
                    consistency_loss=None,
                    ):
        translation_preds = preds_dict["translation_preds"]
        rotation_preds = preds_dict["rotation_preds"]
        if not isinstance(translation_preds, (list, tuple)):
            translation_preds = [translation_preds]
        if not isinstance(rotation_preds, (list, tuple)):
            rotation_preds = [rotation_preds]
        dtype = translation_preds[0].dtype

        pyramid_preds = preds_dict["pyramid_motion"]
        example['icp_odometry'] = example['icp_odometry'].view(-1, 7)

        translation_targets = example['icp_odometry'][:, :3]
        rotation_targets = example['icp_odometry'][:, 3:]

        if translation_loss._loss_weight == 0:
            self.warm_flag = True

        if self.warm_flag:
            if self.get_global_step()<1500:
                warm_weight = 1.0/(0.001*self.get_global_step()+1)
            else:
                warm_weight=0
            translation_loss._loss_weight = warm_weight
            rotation_loss._loss_weight = warm_weight
        else:
            warm_weight = 0

        # consistency loss
        C_loss = torch.zeros([1], dtype=dtype).cuda()
        res_r, res_t = None, None
        if consistency_loss is not None:
            point_confs=None
            if len(preds_dict["middle_conf_preds"])>0:
                #process points
                if "normal_gt" in example:
                    points=[[ torch.cat( [p[:, :3  ], example["normal_gt"][i] ], dim=-1 ) [None] ]
                            for i, p in enumerate(preds_dict["voxel_features"]) 
                            ] #Nx7
                else:
                    if preds_dict["voxel_features"][0].shape[1]>6:
                        points=[[ p[:, [0, 1, 2, 4, 5, 6]][None] ]
                            for p in preds_dict["voxel_features"]] #Nx7
                    else:
                        points=[[ p[:, [0, 1, 2, 3, 4, 5]][None] ]
                            for p in preds_dict["voxel_features"]] #Nx7
                    


                point_confs = [p.reshape(-1,3,3)[None] if p.shape[-1]==9 else p[None]  for p in preds_dict["middle_conf_preds"] ]
                min_len=1e10
                for p in points:
                    min_len=min(p[0].shape[1], min_len) 
                points=[ [p[0][:,:min_len] ]
                      for p in points]
                point_confs = [p[:,:min_len] for p in point_confs]
                point_confs=create_cycle_constraint_data (point_confs)
            else:
                points = example["hier_points"]

            new_points = []  # [h][t]
            for h, _ in enumerate(points[0]):
                points_seq = [points[t][h] for t in range(len(points))]
                new_points.append(create_cycle_constraint_data(
                    points_seq, 1))   # list of BxNx3

            if len(new_points) < len(rotation_preds):
                new_points = new_points + \
                    [new_points[-1]] * (len(rotation_preds) - len(new_points))
            else:
                # select the points with highest
                new_points = new_points[:len(rotation_preds)]


            weights = [0.01, 0.01, 0.05, 0.1, 1]
            for i, (R_pred, T_pred, weight) in enumerate(zip(rotation_preds, translation_preds, weights[-len(translation_preds):])):
                if R_pred.shape[-1] == 9:
                    R_pred = R_pred.reshape(-1, 3, 3)
                else:
                    R_pred = kornia.quaternion_to_rotation_matrix(
                        torchplus.roll(R_pred, shift=-1, dim=-1))  # Bx3x3
                if self.get_global_step()<=1500:
                    R_pred=torch.stack([torch.eye(3,device=R_pred.device)]*R_pred.shape[0], dim=0)
                    T_pred=torch.zeros_like(T_pred)

                # Bx1x3x3 @ BxNx3x1 + Bx1x3x1-> BxNx3x1
                transformed_p1_gt = torch.matmul(
                    R_pred[:, None], new_points[-(i+1)][1][:, :, :3][..., None, ]) + T_pred[:, None, :, None]

                transformed_p1 = new_points[-(i+1)][0][:, :, :3][..., None]

                transformed_normal1_gt = torch.matmul(
                    R_pred[:, None].detach(), new_points[-(i+1)][1][:, :, 3:][..., None])
                transformed_normal1 = new_points[-(i+1)
                                                    ][0][:, :, 3:][..., None]

                if self.get_global_step() > 1500:
                    icp_iter=self.icp_iter
                else:
                    icp_iter=5
                res_r_ = torch.eye(3, device=R_pred.device)
                res_t_ = torch.zeros([3], device=R_pred.device)
                # l_ = 0

                l, res_r, res_t = consistency_loss(transformed_p1.squeeze(-1),
                                                transformed_p1_gt.squeeze(-1), 
                                                cov_pred=point_confs[0], 
                                                cov_target=point_confs[1], 
                                                R_pred=R_pred, t_pred=T_pred,
                                                normal_pred=transformed_normal1.squeeze(-1).detach(), normal_target=transformed_normal1_gt.squeeze(-1).detach(),
                                                mask=None,
                                                icp_iter=icp_iter)

                transformed_p1_gt =  torch.matmul(
                        res_r[:, None], transformed_p1_gt) + res_t[:, None, :, None]
                transformed_normal1_gt = torch.matmul(
                    res_r[:, None].detach(),transformed_normal1_gt)
                #Bx3x3 = Bx3x3@Bx3x3
                res_r_ = res_r@res_r_
                #Bx3 = (Bx3x3@Bx3x1+Bx3x1).squeeze(-1)
                # print (res_r.shape, res_t_.shape,res_t.shape, flush=True)
                res_t_ = (res_r@res_t_[...,None]+res_t[...,None]).squeeze(-1)
                # l_+=l

                res_r = res_r_
                res_t = res_t_ 
                # l = l_
                
                C_loss += (1-warm_weight)*weight*l
                
              
        if res_r is not None and res_t is not None:
            rotation_targets = res_r@R_pred.detach()  # Bx3x3
            rotation_targets = kornia.rotation_matrix_to_quaternion(
                rotation_targets)
            rotation_targets = torchplus.roll(rotation_targets, 1, dim=-1)
            rotation_targets *= torch.sign(rotation_targets[:, 0:1])

            translation_targets = (
                res_r@T_pred[..., None].detach()+res_t[..., None]).squeeze(-1)  # Bx3

        if len(pyramid_preds) > 0:
            tq_map_targets = self.gen_tq_maps(torch.cat([translation_targets, rotation_targets], dim=-1).reshape(-1, 7),
                                              spatial_size=pyramid_preds[-1][0].shape[2:],
                                              pc_range=self.odom_predictor.point_cloud_range, cubic_tq_map=self.odom_predictor._cubic_pred_height > 0)
            example['tq_maps'] = tq_map_targets

        pyramid_targets = [example['tq_maps'][i]
                           for i in range(len(example['tq_maps']))]

        if not isinstance(translation_preds, (list, tuple)):
            translation_preds = [translation_preds]
        T_loss = 0

        for p in translation_preds:
            T_loss += translation_loss(
                p, translation_targets)

        if not isinstance(rotation_preds, (list, tuple)):
            rotation_preds = [rotation_preds]
        R_loss = 0

        for p in rotation_preds:
            R_loss += rotation_loss(
                p, rotation_targets)

        if pyramid_translation_loss is None or pyramid_rotation_loss is None:
            return T_loss, R_loss

        # # pyramid loss
        pyramid_T_losses = []
        pyramid_R_losses = []
        for i, _ in enumerate(pyramid_preds):
            if isinstance(pyramid_preds[i], (tuple, list)):
                T_pred = pyramid_preds[i][0][:, :3]
                R_pred = pyramid_preds[i][0][:, 3:]
                pred_mask = pyramid_preds[i][1]
            else:
                T_pred = pyramid_preds[i][:, :3]
                R_pred = pyramid_preds[i][:, 3:]
                pred_mask = None
            T_target = pyramid_targets[0][:, :3]

            if T_target.shape != T_pred.shape:
                T_target = F.interpolate(
                    T_target, size=T_pred[0, 0].shape, mode='nearest')

            R_target = pyramid_targets[0][:, 3:]

            if R_target.shape != R_pred.shape:
                R_target = F.interpolate(
                    R_target, size=T_pred[0, 0].shape, mode='nearest')

            pyramid_T_losses.append(pyramid_translation_loss(
                T_pred, T_target, mask=pred_mask[:, :1])
            )

            pyramid_R_losses.append(pyramid_rotation_loss(
                R_pred, R_target, mask=pred_mask[:, -1:])
            )

        # return T_loss, R_loss, pyramid_T_losses, pyramid_R_losses, [], [], C_loss#+NormalLoss
        return T_loss, R_loss, pyramid_T_losses, pyramid_R_losses, C_loss#+NormalLoss

def create_cycle_constraint_data(xs, cat_dim=1):
    '''xs: a list of x in size of (N,C,H,W)'''
    assert len(xs) >= 2
    seq_len = len(xs)
    input_shape = xs[0].shape
    x1 = []
    x2 = []
    for i in range(0, seq_len):
        for j in range(i+1, seq_len):
            x1.append(xs[i])
            x2.append(xs[j])

    x1 = torch.stack(x1, dim=cat_dim).reshape(-1, *input_shape[1:])
    x2 = torch.stack(x2, dim=cat_dim).reshape(-1, *input_shape[1:])

    # print(x1.shape, x2.shape, '!!!')
    return [x1, x2]


def create_consistency_loss(
    features1,
    features2,
    tq_map_preds,
    pc_range,
    consistency_loss,
    loss_mask=None,
):
    warped_features2, valid_point = inverse_warp(
        features1, features2, tq_map, pc_range, padding_mode='zeros')
    if loss_mask is None:
        loss_mask = torch.ones_like(valid_point)
    loss = consistency_loss(features1, warped_features2,
                            mask=valid_point*loss_mask)

    return loss
