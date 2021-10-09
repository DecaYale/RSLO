"""Classification and regression loss functions for object detection.

Localization losses:
 * WeightedL2LocalizationLoss
 * WeightedSmoothL1LocalizationLoss

Classification losses:
 * WeightedSigmoidClassificationLoss
 * WeightedSoftmaxClassificationLoss
 * BootstrappedSigmoidClassificationLoss
"""
from abc import ABCMeta, abstractmethod
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchplus

import kornia
import rslo.utils.pose_utils as pose_utils
import apex.amp as amp
def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=np.float32):
    """Creates dense vector with indices set to specific value and rest to zeros.

    This function exists because it is unclear if it is safe to use
      tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

    Args:
      indices: 1d Tensor with integer indices which are to be set to
          indices_values.
      size: scalar with size (integer) of output Tensor.
      indices_value: values of elements specified by indices in the output vector
      default_value: values of other elements in the output vector.
      dtype: data type.

    Returns:
      dense 1D Tensor of shape [size] with indices set to indices_values and the
          rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense


# class Loss(object):
class Loss(nn.Module):
    """Abstract base class for loss functions."""
    __metaclass__ = ABCMeta

    def __init__(self, loss_weight=1):
        super(Loss, self).__init__()
        self._loss_weight = loss_weight

    # def __call__(self,
    def forward(self,
                prediction_tensor,
                target_tensor,
                ignore_nan_targets=False,
                scope=None,
                **params):
        """Call the loss function.

        Args:
          prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
            representing predicted quantities.
          target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
            regression or classification targets.
          ignore_nan_targets: whether to ignore nan targets in the loss computation.
            E.g. can be used if the target tensor is missing groundtruth data that
            shouldn't be factored into the loss.
          scope: Op scope name. Defaults to 'Loss' if None.
          **params: Additional keyword arguments for specific implementations of
                  the Loss.

        Returns:
          loss: a tensor representing the value of the loss function.
        """
        if ignore_nan_targets:
            target_tensor = torch.where(torch.isnan(target_tensor),
                                        prediction_tensor,
                                        target_tensor)
        ret = self._compute_loss(prediction_tensor, target_tensor, **params)
        if isinstance(ret, (list, tuple)):
            return [self._loss_weight*ret[0]] + list(ret[1:])
        else:
            return self._loss_weight*self._compute_loss(prediction_tensor, target_tensor, **params)

    @abstractmethod
    @amp.float_function
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overridden by implementations.

        Args:
          prediction_tensor: a tensor representing predicted quantities
          target_tensor: a tensor representing regression or classification targets
          **params: Additional keyword arguments for specific implementations of
                  the Loss.

        Returns:
          loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
            anchor
        """
        pass


class L2Loss(Loss):

    def __init__(self, loss_weight=1):
        super(L2Loss, self).__init__(loss_weight)

    def _compute_loss(self, prediction_tensor, target_tensor, mask=None):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor

        if mask is not None:
            mask = mask.expand_as(diff).byte()
            diff = diff[mask]
        # square_diff = 0.5 * weighted_diff * weighted_diff
        square_diff = diff * diff
        return square_diff.mean()


class AdaptiveWeightedL2Loss(Loss):

    def __init__(self, init_alpha, learn_alpha=True, loss_weight=1, focal_gamma=0, balance_scale=1):
        super(AdaptiveWeightedL2Loss, self).__init__(loss_weight)
        self.learn_alpha = learn_alpha
        # self.balance_scale=balance_scale
        self.alpha = nn.Parameter(torch.Tensor(
            [init_alpha]), requires_grad=learn_alpha)
        self.focal_gamma = focal_gamma
        # self.alpha_shift = -13  # -10# TODO: temporarily test

    def _compute_loss(self, prediction_tensor, target_tensor, mask=None, alpha=None, focal_gamma=None):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """


        if focal_gamma is None:
            focal_gamma = self.focal_gamma
        _alpha = self.alpha
        if mask is None:
            mask = torch.ones_like(target_tensor)
        else:
            mask = mask.expand_as(target_tensor)

        diff = prediction_tensor - target_tensor
        # square_diff = 0.5 * weighted_diff * weighted_diff
        # square_diff = (diff * diff) * mask

        square_diff = (diff * diff) * mask

        # loss = square_diff.mean()
        input_shape = prediction_tensor.shape
        loss = torch.sum(square_diff, dim=list(range(1, len(input_shape)))) / \
            (torch.sum(mask, dim=list(range(1, len(input_shape)))) + 1e-12)  # (B,)


        focal_weight = (torch.exp(-_alpha) * loss)**focal_gamma
        # focal_weight = focal_weight.detach()
        focal_weight = focal_weight/(torch.sum(focal_weight) + 1e-12)
        # focal_weight = focal_weight.detach()

        loss = focal_weight*(torch.exp(-_alpha) * loss)
        loss = loss.sum() + _alpha#*self.balance_scale
        return loss


class AdaptiveWeightedL2RMatrixLoss(Loss):

    def __init__(self, init_alpha, learn_alpha=True, loss_weight=1, focal_gamma=0):
        super(AdaptiveWeightedL2RMatrixLoss, self).__init__(loss_weight)
        self.learn_alpha = learn_alpha
        self.alpha = nn.Parameter(torch.Tensor(
            [init_alpha]), requires_grad=learn_alpha)
        self.focal_gamma = focal_gamma
        # self.alpha_shift = -13  # -10# TODO: temporarily test

    def _compute_loss(self, prediction_tensor, target_tensor, mask=None, alpha=None, focal_gamma=None):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """

        if focal_gamma is None:
            focal_gamma = self.focal_gamma
        _alpha = self.alpha

        if len(target_tensor.shape) == 4:  # map
            target_tensor = target_tensor.permute(0, 2, 3, 1)
        origin_tgt_shape = target_tensor.shape
        
        if target_tensor.shape[-1] == 4:  # quaternion
            target_tensor = torchplus.roll(target_tensor, shift=-1, dim=-1)
            target_tensor = kornia.quaternion_to_rotation_matrix(target_tensor)

        if len(prediction_tensor.shape) == 4:  # map
            prediction_tensor = prediction_tensor.permute(0, 2, 3, 1)
          

        # assert prediction_tensor.shape[-1] ==9
        if prediction_tensor.shape[-1] == 4:  # quaternion
            prediction_tensor = torchplus.roll(
                prediction_tensor, shift=-1, dim=-1)
            prediction_tensor = kornia.quaternion_to_rotation_matrix(
                prediction_tensor)
            
        elif prediction_tensor.shape[-1] == 9:  # rotation matrix vector
            prediction_tensor = prediction_tensor.reshape(-1, 3, 3)
        else:
            raise ValueError

        if mask is None:
            mask = torch.ones_like(target_tensor)
        else:
            if len(mask.shape) == 4:  # map
                mask = mask.permute(0, 2, 3, 1)
            mask = mask[...,None].expand(*origin_tgt_shape[:-1], 3, 3).reshape(target_tensor.shape)
            # mask = mask.expand_as(target_tensor)

        diff = torch.matmul(
            prediction_tensor.transpose(-1, -2), target_tensor) - torch.eye(3, device=target_tensor.device)  # BxHxWx3x3

        # loss = torch.norm(diff-torch.eye(3, device=diff.device).expand_as(diff), dim=0 )**2
        square_diff = (diff * diff) * mask
        loss = torch.sum(square_diff, dim=list(range(
            1, len(diff.shape)))) / (torch.sum(mask, dim=list(range(1, len(diff.shape)))) + 1e-12)


        # focal_weight=1
        focal_weight = (torch.exp(-_alpha) * loss)**focal_gamma
        # focal_weight = focal_weight.detach()
        focal_weight = focal_weight/(torch.sum(focal_weight) + 1e-12)
        # focal_weight = focal_weight.detach()

        loss = focal_weight*(torch.exp(-_alpha) * loss)
        loss = loss.sum() + _alpha
        return loss


class ChamferL2Loss(Loss):
    def __init__(self, init_alpha=0, learn_alpha=False, loss_weight=1, focal_gamma=0, n_samples=7000, penalize_ratio=0.5, sample_block_size=(0.1,1,1)):
        super(ChamferL2Loss, self).__init__(loss_weight)

        from thirdparty.chamfer_distance.chamfer_distance import ChamferDistance
        self.learn_alpha = learn_alpha
        self.alpha = nn.Parameter(torch.Tensor(
            [init_alpha]), requires_grad=learn_alpha)
        self.focal_gamma = focal_gamma
        self.n_samples=n_samples
        self.penalize_ratio = penalize_ratio
        self.sample_block_size = sample_block_size
        self.cd = ChamferDistance()

    





# class Aleat5_1ChamferL2NormalWeightedALLSVDLoss(ChamferL2Loss):
class Aleat5_1ChamferL2NormalWeightedALLSVDLoss(Loss):
    #aleat_covariance+range+chamfer loss  +reg_weight + double conf
    def __init__(self, init_alpha=0, learn_alpha=False, loss_weight=1, focal_gamma=0, n_samples=-1, penalize_ratio=0.95, sample_block_size=(0.1,1,1), norm=True, pred_downsample_ratio=1, reg_weight=0.001, sph_weight=1):
        # super().__init__(init_alpha=init_alpha, learn_alpha=learn_alpha, loss_weight=loss_weight, focal_gamma=focal_gamma, n_samples=n_samples, penalize_ratio=penalize_ratio, sample_block_size=sample_block_size)
        super().__init__(loss_weight=loss_weight)

        self.learn_alpha = learn_alpha
        self.alpha = nn.Parameter(torch.Tensor(
            [init_alpha]), requires_grad=learn_alpha)
        self.focal_gamma = focal_gamma
        self.n_samples=n_samples
        self.penalize_ratio = penalize_ratio
        self.sample_block_size = sample_block_size

        # from thirdparty.chamfer_distance.chamfer_distance import ChamferDistanceWithIdx
        from thirdparty.chamfer_distance.chamfer_distance import OneDirectionChamferDistanceWithIdx
        from rslo.layers.svd import SVDHead
        # self.cd = ChamferDistanceWithIdx()
        self.cd = OneDirectionChamferDistanceWithIdx()
        self.norm = norm
        self.svd = SVDHead()
        self.pred_downsample_ratio = pred_downsample_ratio
        self.reg_weight=reg_weight
        self.sph_weight = sph_weight

    def _points_roi(self, dist, penalize_ratio=0.95, dist_threshold=None ):
        #dist: (N,)
        if dist_threshold is None:
            m,_ = torch.kthvalue(dist.reshape(-1), 1+int(len(dist.reshape(-1)-1 )*penalize_ratio), dim=-1)
            m = torch.max(m, torch.ones_like(m))
            roi_mask = dist<m
        else:
            roi_mask = dist<dist_threshold
        return roi_mask

    # def _compute_loss(self, xyz_pred, xyz_target, normal_pred, normal_target, mask=None, alpha=None, focal_gamma=None):
    def _compute_loss(self, xyz_pred, xyz_target, cov_pred, cov_target, R_pred, t_pred,normal_pred, normal_target, mask=None, alpha=None, focal_gamma=None, icp_iter=1):
        """Compute loss function.
        Args: 
            xyz_pred: BxNx3
            normal_pred: BxNx3
            cov_pred: BxNx9
            cov_target: BxMx9
        Returns:

        """

        def span_cov2(cov_param_pred, return_eig_vec=False):
            #cov_param: Nx7 

            cov_param = cov_param_pred.clone()
            cov_param[:,1:2] = cov_param[:,0:1]+cov_param_pred[:,1:2]#!!
            cov_param[:,2:3] = cov_param[:,1:2]+cov_param_pred[:,2:3]#!!
            cov_param[:,3:]  = cov_param[:,3:].clone() / (torch.norm(cov_param_pred[:,3:], dim=-1, keepdim=True)+1e-9)

            eigval=torch.zeros(cov_param.shape[0],9, device=cov_param.device, dtype=cov_param.dtype)
            eigval[:,::4] = cov_param[:,:3]
            eigval=eigval.reshape(-1,3,3) 
            eigvec=kornia.quaternion_to_rotation_matrix(cov_param[:,3:] ) #Nx3x3
            if not return_eig_vec:
                return eigvec@eigval@eigvec.transpose(-1,-2)
            else:
                return eigvec@eigval@eigvec.transpose(-1,-2), eigvec

        if focal_gamma is None:
            focal_gamma = self.focal_gamma
        _alpha = self.alpha

        if mask is None:
            mask = [torch.ones_like(xyz_pred[...,:1])  ]*2
        else:
            mask = [m.expand_as(xyz_pred[...,:1]) for m in mask]

        if self.pred_downsample_ratio<1:
            selected_inds = np.random.choice(len(xyz_pred[0]),int(len(xyz_pred[0])*self.pred_downsample_ratio), replace=False )
            xyz_pred = xyz_pred[:,selected_inds]
            cov_pred = cov_pred[:,selected_inds]
            mask[0]=mask[0][:,selected_inds]


        xyz_pred = [xyz_pred[b] for b in range(len(xyz_pred))]
        xyz_target = [xyz_target[b] for b in range(len(xyz_target))]
        cov_pred = [cov_pred[b] for b in range(len(cov_pred))]
        cov_target = [cov_target[b] for b in range(len(cov_target))]
        mask[0] = [mask[0][b] for b in range(len(mask[0]))]

        normal_pred = [normal_pred[b] for b in range(len(normal_pred))]
        normal_target = [normal_target[b] for b in range(len(normal_target))]
        

        loss = 0 
        res_R=[]
        res_T=[]
        for b in range(len(xyz_pred) ):
            #span cov 
            cov_pred[b],cov_normal= span_cov2(cov_pred[b], True)
            cov_target[b],_ = span_cov2(cov_target[b], True)

            
            # diff, diff21, idx1, idx2 = self.cd(xyz_pred[b][None,...], xyz_target[b][None,...]) #diff 1xN #1,N,3
            diff, idx1  = self.cd(xyz_pred[b][None,...], xyz_target[b][None,...]) #diff 1xN #1,N,3

            diff=diff.squeeze(0)
            idx1 = idx1.long().squeeze(0)
            xyz_assoc = xyz_target[b][idx1]
            mask_assoc = mask[1][b][idx1]
            cov_pred_assoc=cov_target[b][idx1]

            diff_vec=xyz_pred[b]-xyz_assoc #Nx3

            weight = nn.functional.cosine_similarity(normal_pred[b], xyz_assoc-xyz_pred[b], dim=-1)[...,None].abs()
            
            mask_b  = mask[0][b]#*weight*weight #
            count_mask=self._points_roi(diff[None], penalize_ratio=self.penalize_ratio)

            mask_b = mask_b.squeeze(-1)[count_mask.squeeze(0)]
            mask_assoc_sel = mask_assoc.squeeze(-1)[count_mask.squeeze(0)]
            cov_pred[b] = cov_pred[b][count_mask.squeeze(0)]
            cov_pred_assoc = cov_pred_assoc[count_mask.squeeze(0)]
            diff_vec=diff_vec[count_mask.squeeze(0)]

            sigma=cov_pred[b]+R_pred[b].detach()@cov_pred_assoc@R_pred[b].detach().transpose(-1,-2)
            try:
                sigma_inv = torch.inverse(sigma)
            except:
                try:
                    sigma_inv=torch.inverse(sigma+torch.eye(3,device=sigma.device, dtype=sigma.dtype)*1e-6)
                except:
                    print("Set segma_inv=eye")
                    sigma_inv=torch.eye(3,device=sigma.device, dtype=sigma.dtype)

            square_diff = (diff_vec.unsqueeze(-2)@sigma_inv @ diff_vec[...,None]).squeeze(-1) #N,   

            input_shape = square_diff.shape
            loss_ = torch.mean(square_diff)+self.reg_weight*torch.mean(0.5*torch.log(torch.det(sigma)))

            loss += loss_

            #svd 
            src_mask = count_mask[...,None].expand_as(xyz_pred[b][None,...])
            src = xyz_pred[b][None,...][ src_mask ].reshape(1,-1,3).permute(0,2,1).detach()
            tgt = xyz_assoc[None,...][src_mask].reshape(1,-1,3).permute(0,2,1).detach()
            
            #TODO
            mask_b2=mask[0][b][None].squeeze(-1)[count_mask].reshape(1,-1).detach()
            mask_assoc_sel2=mask_assoc[None].squeeze(-1)[count_mask].reshape(1,-1).detach()
            wgt = weight[None].squeeze(-1)[count_mask].reshape(1,-1).detach() #

            res_r_ = torch.eye(3, device=R_pred.device)
            res_t_ = torch.zeros([3], device=R_pred.device)
            for icp_i in range(icp_iter):
                try:
                    R,t = self.svd(
                        src,
                        tgt,
                        weight=wgt**2 )
                except:
                    print("svd failed", flush=True)
                    R = torch.eye(3, device=wgt.device, dtype=src.dtype)[None]
                    t = torch.zeros([1,3], device=weight.device, dtype=src.dtype)
                
                #Bx3x3 = Bx3x3@Bx3x3
                res_r_ = R@res_r_
                #Bx3 = (Bx3x3@Bx3x1+Bx3x1).squeeze(-1)
                res_t_ = (R@res_t_[...,None]+t[...,None]).squeeze(-1)            
                
                if icp_i < icp_iter-1:
                    #Bx1x3x3 @ BxNx3x1 + Bx1x3x1-> BxNx3x1
                    # tgt =  torch.matmul(
                    #                 R[:, None], xyz_target[b][None,...,None]) + t[:, None, :, None] # 1xNx3x1
                    tgt =  torch.matmul(
                                    res_r_[:, None], xyz_target[b][None,...,None]) + res_t_[:, None, :, None] # 1xNx3x1
                    tgt=tgt.squeeze(0).squeeze(-1)# Nx3

                    #chamfer dist
                    orig_diff, orig_idx1  = self.cd(xyz_pred[b][None,...], tgt[None]) #1xNx1
                    orig_diff=orig_diff.squeeze(0)
                    orig_idx1 = orig_idx1.long().squeeze(0)
                    orig_xyz_assoc = tgt[orig_idx1]
                    wgt =nn.functional.cosine_similarity(normal_pred[b], orig_xyz_assoc-xyz_pred[b], dim=-1)[...,None].abs() #Nx1

                    #compute roi_mask
                    roi_mask = self._points_roi(orig_diff[None], penalize_ratio=self.penalize_ratio) #1xN

                    src_mask = roi_mask[...,None].expand_as(xyz_pred[b][None,...]) #Nx1->1xNx3
                    src = xyz_pred[b][None,...][ src_mask ].reshape(1,-1,3).permute(0,2,1).detach()
                    tgt = orig_xyz_assoc[None,...][src_mask].reshape(1,-1,3).permute(0,2,1).detach()
                    wgt = wgt[None].squeeze(-1)[roi_mask].reshape(1,-1).detach()

            res_R.append(res_r_)
            res_T.append(res_t_)

        res_R=torch.cat(res_R, dim=0)
        res_T=torch.cat(res_T, dim=0)

        loss/=len(xyz_pred)
            
        # focal_weight=1
        focal_weight = (torch.exp(-_alpha) * loss)**focal_gamma
        # focal_weight = focal_weight.detach()
        focal_weight = focal_weight/(torch.sum(focal_weight) + 1e-12)
        # focal_weight = focal_weight.detach()

        loss = focal_weight*(torch.exp(-_alpha) * loss)
        # loss =loss.mean()
        loss = loss.sum() + _alpha
        return loss, res_R, res_T


