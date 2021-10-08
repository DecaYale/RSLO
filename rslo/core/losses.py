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
        print("Balance scale:", balance_scale)
        self.balance_scale=balance_scale
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

        # if alpha is not None:
        #     _alpha = alpha+self.alpha_shift
        # else:
        #     _alpha = self.alpha
        ''' 
        _alpha = self.alpha
        if mask is not None:
            diff = (prediction_tensor -
                    target_tensor)[(mask > 0.5).expand_as(prediction_tensor)]
            if len(diff) == 0:
                diff = torch.zeros([1]).cuda()
        else:
            diff = prediction_tensor - target_tensor
        # square_diff = 0.5 * weighted_diff * weighted_diff
        square_diff = diff * diff
        loss = square_diff.mean()
        # loss = torch.exp(-self.alpha) * loss + self.alpha
        loss = torch.exp(-_alpha) * loss + _alpha
        '''

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
        # loss = loss.mean()

        # loss = torch.exp(-self.alpha) * loss + self.alpha
        # loss = torch.exp(-_alpha) * loss + _alpha
        # focal_weight = (torch.exp(-_alpha) * loss).detach()**focal_gamma

        # focal_weight=1
        focal_weight = (torch.exp(-_alpha) * loss)**focal_gamma
        # focal_weight = focal_weight.detach()
        focal_weight = focal_weight/(torch.sum(focal_weight) + 1e-12)
        # focal_weight = focal_weight.detach()

        # + _alpha / loss.shape[0]
        loss = focal_weight*(torch.exp(-_alpha) * loss)
        # print(focal_gamma, loss,'!!!', loss.shape)
        # loss =loss.mean()
        loss = loss.sum() + _alpha*self.balance_scale
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
        # if mask is None:
        #     mask = torch.ones_like(target_tensor)
        # else:
        #     mask = mask.expand_as(target_tensor)
        # pred_shape = prediction_tensor.shape
        # if len(pred_shape)==4: #map
        #   prediction_tensor = prediction_tensor.permute(0,2,3,1 )#.reshape([-1, ])

        # else: #vector
        #   pass

        if len(target_tensor.shape) == 4:  # map
            target_tensor = target_tensor.permute(0, 2, 3, 1)
        origin_tgt_shape = target_tensor.shape
        
        if target_tensor.shape[-1] == 4:  # quaternion
            # formatting1
            # print(target_tensor,'!')
            # buf = target_tensor[...,0].clone()
            # target_tensor[...,:3] = target_tensor[...,1:].clone()
            # target_tensor[...,-1] = buf
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

        # pred_shape = prediction_tensor.shape
        # if len(pred_shape)==4: #map
        #   prediction_tensor = prediction_tensor.permute(0,2,3,1 )#.reshape([-1, ])

        diff = torch.matmul(
            prediction_tensor.transpose(-1, -2), target_tensor) - torch.eye(3, device=target_tensor.device)  # BxHxWx3x3

        # loss = torch.norm(diff-torch.eye(3, device=diff.device).expand_as(diff), dim=0 )**2
        square_diff = (diff * diff) * mask
        loss = torch.sum(square_diff, dim=list(range(
            1, len(diff.shape)))) / (torch.sum(mask, dim=list(range(1, len(diff.shape)))) + 1e-12)

        # square_diff = (diff * diff) * mask

        # loss = square_diff.mean()
        # input_shape = prediction_tensor.shape
        # loss = torch.sum(square_diff, dim=list(range(1, len(input_shape)))) / \
        #     (torch.sum(mask, dim=list(range(1, len(input_shape))) )  +1e-12)# (B,)

        # focal_weight=1
        focal_weight = (torch.exp(-_alpha) * loss)**focal_gamma
        # focal_weight = focal_weight.detach()
        focal_weight = focal_weight/(torch.sum(focal_weight) + 1e-12)
        # focal_weight = focal_weight.detach()

        # + _alpha / loss.shape[0]
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

    def _random_block_choose(self, points_list, mask, block_size=(0.1,1,1)):
        def where(cond, x_1, x_2):
            cond = cond.float()    
            return (cond * x_1) + ((1-cond) * x_2)

        def get_boundary(points, margin=0.05):
            bounds=[]
            for i in range(points.shape[-1]): #BxNx3
                bounds.append( [torch.min(points[:,:,i], dim=1)[0], torch.max(points[:, :, i], dim=1)[0]]) #(B,)
            
            for i in range(len(bounds)):
                width = bounds[i][1] - bounds[i][0]
                bounds[i][0] += width*margin
                bounds[i][1] -= width*margin

            return bounds

        def select_block(block_size):
            dim_size = [1/s for s in block_size]
            # n  = dim_size[0] * dim_size[1] * dim_size[2]
            i = np.random.randint(dim_size[0])#*block_size[0]
            j = np.random.randint(dim_size[1])#*block_size[1]
            k = np.random.randint(dim_size[2])#*block_size[2]

            return i,j,k # 
        def get_block_range(i,j,k, bounds, block_size):
            ranges=[]
            for p, r in enumerate([i,j,k]):
                _min = (bounds[p][1]- bounds[p][0])* r*block_size[p] + bounds[p][0] #(B,)
                _max = _min+ (bounds[p][1]- bounds[p][0])*block_size[p]
                ranges.append([_min, _max])
            return ranges #in meters, with exclusive higher bound 

        bounds = get_boundary(points_list[0]) # [(B,)(B,),(B,)] 
        # print("bounds:", bounds)
        i_,j_,k_ = select_block(block_size)
        # print("i,jk:", i,j,k)
        selected_range = get_block_range(i_,j_,k_, bounds, block_size) 
        # print("selected_range", selected_range)


        indicator_list = []
        
        for i, points in enumerate(points_list):
            # print(points.shape, selected_range, len(selected_range[0]) )
            # indicator = where( ( (selected_range[0][0][:,None]<points[:,:,0]).float() 
            #   +(points[:,:,0]<selected_range[0][1][:,None] ).float() 
            #      +(selected_range[1][0][:,None]<points[:,:,1]).float()
            #      +(points[:,:,1]<selected_range[1][1][:,None]).float() 
            #     +(selected_range[2][0][:,None]<points[:,:,2]).float()
            #     +(points[:,:,2]<selected_range[2][1][:,None] ).float() ).detach()>5.5 , torch.ones_like(points[:,:,0]), torch.zeros_like(points[:,:,0]) ) #BxN #x3?
            indicator = where( ( (selected_range[0][0][:,None]<points[:,:,0])
              &(points[:,:,0]<selected_range[0][1][:,None] )
                 &(selected_range[1][0][:,None]<points[:,:,1])
                 &(points[:,:,1]<selected_range[1][1][:,None]) 
                &(selected_range[2][0][:,None]<points[:,:,2])
                &(points[:,:,2]<selected_range[2][1][:,None] ) ) , torch.ones_like(points[:,:,0]), torch.zeros_like(points[:,:,0]) ) #BxN #x3?

            indicator_list.append(indicator)
            # print(indicator_list[i].shape)

        #split in batch dimmension
        new_points_list = []
        for i in range(len(indicator_list)):
            batch=[]
            for b in range(indicator_list[0].shape[0]):
                # import pdb 
                # pdb.set_trace()
                inds = indicator_list[i][b].nonzero()
                if len(inds)<500:
                    # print("points_list[i][b][0]", points_list[i][b][0])
                    if i==0:
                        batch.append(points_list[i][b][0:1] ) 
                    else:
                        batch.append(points_list[i][b] ) 

                else:
                    batch.append(torch.index_select(points_list[i][b].clone(), 0, inds.squeeze(1) ) ) 
                    # print(len(inds),points_list[i][b].shape,flush=True)
                # batch.append(points_list[i][b]) 
            
            new_points_list.append(batch)
        new_mask=[]
        for b in range(mask.shape[0]):
            inds = indicator_list[0][b].nonzero()
            # print(torch.max(inds), mask[b].shape)
            if len(inds)<500:
                new_mask.append(mask[b][0:1]) 
            else:
                new_mask.append(torch.index_select(mask[b].clone(), 0, inds.squeeze(1) ) ) 
            # new_mask.append(mask[b]) 

        return new_points_list[0], new_points_list[1], new_mask 
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
            mask = torch.ones_like(prediction_tensor[...,:1])
        else:
            mask = mask.expand_as(prediction_tensor[...,:1])

        # diff = prediction_tensor - target_tensor
        # select_indices= torch.arange(min(prediction_tensor.shape[1], self.n_samples) )
        
        # select_indices = select_indices[torch.randint(0, select_indices.size(0), (min(self.n_samples, select_indices.size(0) ),) ) ]
        # select_indices= torch.randperm(prediction_tensor.shape[1])
        # select_indices = select_indices[ :min(self.n_samples, len(select_indices) ) ]
        # prediction_tensor = prediction_tensor[:,select_indices]
        # target_tensor = target_tensor[:,select_indices]
        # mask=mask[:,select_indices]

        prediction_tensor, target_tensor,mask = self._random_block_choose([prediction_tensor, target_tensor], mask=mask, block_size=self.sample_block_size)

        loss = 0 
        # import pdb 
        # pdb.set_trace()
        for b in range(len(prediction_tensor) ):


            diff, _ = self.cd(prediction_tensor[b][None,...], target_tensor[b][None,...])
            
            # print(diff.shape, int(len(diff.reshape(-1))*self.penalize_ratio), flush=True)
            m,_ = torch.kthvalue(diff.reshape(-1), 1+int(len(diff.reshape(-1)-1 )*self.penalize_ratio), dim=-1)
            # m = diff.median()
            diff_ = diff[diff<m]
            mask[b] = mask[b].squeeze(-1)[diff.squeeze(0)<m]
            # print(len(diff_), '!!diff')

            square_diff = (diff_ * diff_) * mask[b]

            # loss = square_diff.mean()
            input_shape = square_diff.shape
            loss_ = torch.sum(square_diff, dim=list(range(1, len(input_shape)))) / \
                (torch.sum(mask[b], dim=list(range(1, len(input_shape)))) + 1e-12)  # (B,)
            loss += loss_

        loss/=len(prediction_tensor)
            
         
        # if mask is None:
        #     mask = torch.ones_like(diff)
        # else:
        #     mask = mask.expand_as(diff)

       
        # loss = loss.mean()

        # loss = torch.exp(-self.alpha) * loss + self.alpha
        # loss = torch.exp(-_alpha) * loss + _alpha
        # focal_weight = (torch.exp(-_alpha) * loss).detach()**focal_gamma

        # focal_weight=1
        focal_weight = (torch.exp(-_alpha) * loss)**focal_gamma
        # focal_weight = focal_weight.detach()
        focal_weight = focal_weight/(torch.sum(focal_weight) + 1e-12)
        # focal_weight = focal_weight.detach()

        # + _alpha / loss.shape[0]
        loss = focal_weight*(torch.exp(-_alpha) * loss)
        # print(focal_gamma, loss,'!!!', loss.shape)
        # loss =loss.mean()
        loss = loss.sum() + _alpha
        return loss



class ChamferL2NormalWeightedLoss(ChamferL2Loss):
    
    def __init__(self, init_alpha=0, learn_alpha=False, loss_weight=1, focal_gamma=0, n_samples=7000, penalize_ratio=0.5, sample_block_size=(0.1,1,1)):
        super().__init__(init_alpha=init_alpha, learn_alpha=learn_alpha, loss_weight=loss_weight, focal_gamma=focal_gamma, n_samples=n_samples, penalize_ratio=penalize_ratio, sample_block_size=sample_block_size)

        from thirdparty.chamfer_distance.chamfer_distance import ChamferDistanceWithIdx
        self.cd = ChamferDistanceWithIdx()
    def _random_block_choose(self, points_list, normal_list, mask, block_size=(0.1,1,1)):
        def where(cond, x_1, x_2):
            cond = cond.float()    
            return (cond * x_1) + ((1-cond) * x_2)

        def get_boundary(points, margin=0.05):
            bounds=[]
            for i in range(points.shape[-1]): #BxNx3
                bounds.append( [torch.min(points[:,:,i], dim=1)[0], torch.max(points[:, :, i], dim=1)[0]]) #(B,)
            
            for i in range(len(bounds)):
                width = bounds[i][1] - bounds[i][0]
                bounds[i][0] += width*margin
                bounds[i][1] -= width*margin

            return bounds

        def select_block(block_size):
            dim_size = [1/s for s in block_size]
            # n  = dim_size[0] * dim_size[1] * dim_size[2]
            i = np.random.randint(dim_size[0])#*block_size[0]
            j = np.random.randint(dim_size[1])#*block_size[1]
            k = np.random.randint(dim_size[2])#*block_size[2]

            return i,j,k # 
        def get_block_range(i,j,k, bounds, block_size):
            ranges=[]
            for p, r in enumerate([i,j,k]):
                _min = (bounds[p][1]- bounds[p][0])* r*block_size[p] + bounds[p][0] #(B,)
                _max = _min+ (bounds[p][1]- bounds[p][0])*block_size[p]
                ranges.append([_min, _max])
            return ranges #in meters, with exclusive higher bound 

        bounds = get_boundary(points_list[0]) # [(B,)(B,),(B,)] 
        # print("bounds:", bounds)
        i_,j_,k_ = select_block(block_size)
        # print("i,jk:", i,j,k)
        selected_range = get_block_range(i_,j_,k_, bounds, block_size) 
        # print("selected_range", selected_range)


        indicator_list = []
        
        for i, points in enumerate(points_list):
            # print(points.shape, selected_range, len(selected_range[0]) )
            # indicator = where( ( (selected_range[0][0][:,None]<points[:,:,0]).float() 
            #   +(points[:,:,0]<selected_range[0][1][:,None] ).float() 
            #      +(selected_range[1][0][:,None]<points[:,:,1]).float()
            #      +(points[:,:,1]<selected_range[1][1][:,None]).float() 
            #     +(selected_range[2][0][:,None]<points[:,:,2]).float()
            #     +(points[:,:,2]<selected_range[2][1][:,None] ).float() ).detach()>5.5 , torch.ones_like(points[:,:,0]), torch.zeros_like(points[:,:,0]) ) #BxN #x3?
            indicator = where( ( (selected_range[0][0][:,None]<points[:,:,0])
              &(points[:,:,0]<selected_range[0][1][:,None] )
                 &(selected_range[1][0][:,None]<points[:,:,1])
                 &(points[:,:,1]<selected_range[1][1][:,None]) 
                &(selected_range[2][0][:,None]<points[:,:,2])
                &(points[:,:,2]<selected_range[2][1][:,None] ) ) , torch.ones_like(points[:,:,0]), torch.zeros_like(points[:,:,0]) ) #BxN #x3?

            indicator_list.append(indicator)
            # print(indicator_list[i].shape)

        #split in batch dimmension
        new_points_list = []
        new_normal_list = []
        for i in range(len(indicator_list)):
            batch=[]
            normal_batch=[]
            for b in range(indicator_list[0].shape[0]):
                # import pdb 
                # pdb.set_trace()
                inds = indicator_list[i][b].nonzero()
                if len(inds)<500:
                    # print("points_list[i][b][0]", points_list[i][b][0])
                    if i==0:
                        batch.append(points_list[i][b][0:1] ) 
                        normal_batch.append(normal_list[i][b][0:1] )
                    else:
                        batch.append(points_list[i][b] ) 
                        normal_batch.append(normal_list[i][b] )

                else:
                    batch.append(torch.index_select(points_list[i][b].clone(), 0, inds.squeeze(1) ) ) 
                    normal_batch.append(torch.index_select(normal_list[i][b].clone(), 0, inds.squeeze(1) ) ) 
                    # print(len(inds),points_list[i][b].shape,flush=True)
                # batch.append(points_list[i][b]) 
            new_points_list.append(batch)
            new_normal_list.append(normal_batch)

        new_mask=[]
        for b in range(mask.shape[0]):
            inds = indicator_list[0][b].nonzero()
            # print(torch.max(inds), mask[b].shape)
            if len(inds)<500:
                new_mask.append(mask[b][0:1]) 
            else:
                new_mask.append(torch.index_select(mask[b].clone(), 0, inds.squeeze(1) ) ) 
            # new_mask.append(mask[b]) 
        
        

        return new_points_list[0], new_points_list[1], new_normal_list[0], new_normal_list[1],new_mask 
    def _compute_loss(self, xyz_pred, xyz_target, normal_pred, normal_target, mask=None, alpha=None, focal_gamma=None):
        """Compute loss function.
        Args: 
            xyz_pred: BxNx3
            normal_pred: BxNx3
        Returns:

        """

        if focal_gamma is None:
            focal_gamma = self.focal_gamma
        _alpha = self.alpha

        if mask is None:
            mask = torch.ones_like(xyz_pred[...,:1])
        else:
            mask = mask.expand_as(xyz_pred[...,:1])

        xyz_pred, xyz_target, normal_pred, normal_target,mask = self._random_block_choose([xyz_pred, xyz_target], normal_list=[normal_pred, normal_target],mask=mask, block_size=self.sample_block_size)

        loss = 0 
        for b in range(len(xyz_pred) ):
            diff, diff21, idx1, idx2 = self.cd(xyz_pred[b][None,...], xyz_target[b][None,...]) #diff 1xN
            idx1 = idx1.long().squeeze(0)
            xyz_assoc = xyz_target[b][idx1]
            weight = nn.functional.cosine_similarity(normal_pred[b], xyz_assoc-xyz_pred[b], dim=-1).abs()
            diff = diff*weight
            # nn_normal1 = normal_target[b, idx1].squeeze(-1) #Nx3

            # print(diff.shape, idx1.shape, idx1.dtype, normal_target.shape, nn_normal1.shape,'!!!', flush=True)
            
            # print(diff.shape, int(len(diff.reshape(-1))*self.penalize_ratio), flush=True)
            m,_ = torch.kthvalue(diff.reshape(-1), 1+int(len(diff.reshape(-1)-1 )*self.penalize_ratio), dim=-1)
            # m = diff.median()
            count_mask = diff<m
            diff_ = diff[count_mask]
            mask[b] = mask[b].squeeze(-1)[diff.squeeze(0)<m]
            # nn_normal1 = nn_normal1[count_mask[...,None].squeeze(0).expand_as(nn_normal1) ].reshape(-1,3)

            square_diff = (diff_ * diff_) * mask[b]  

            # loss = square_diff.mean()
            input_shape = square_diff.shape
            loss_ = torch.sum(square_diff, dim=list(range(1, len(input_shape)))) / \
                (torch.sum(mask[b], dim=list(range(1, len(input_shape)))) + 1e-12)  # (B,)
            loss += loss_

        loss/=len(xyz_pred)
            

        # focal_weight=1
        focal_weight = (torch.exp(-_alpha) * loss)**focal_gamma
        # focal_weight = focal_weight.detach()
        focal_weight = focal_weight/(torch.sum(focal_weight) + 1e-12)
        # focal_weight = focal_weight.detach()

        # + _alpha / loss.shape[0]
        loss = focal_weight*(torch.exp(-_alpha) * loss)
        # print(focal_gamma, loss,'!!!', loss.shape)
        # loss =loss.mean()
        loss = loss.sum() + _alpha
        return loss




class Aleat5_1ChamferL2NormalWeightedALLSVDLoss(ChamferL2NormalWeightedLoss):
    #aleat_covariance+range+chamfer loss  +reg_weight + double conf
    def __init__(self, init_alpha=0, learn_alpha=False, loss_weight=1, focal_gamma=0, n_samples=-1, penalize_ratio=0.95, sample_block_size=(0.1,1,1), norm=True, pred_downsample_ratio=1, reg_weight=0.001, sph_weight=1):
        super().__init__(init_alpha=init_alpha, learn_alpha=learn_alpha, loss_weight=loss_weight, focal_gamma=focal_gamma, n_samples=n_samples, penalize_ratio=penalize_ratio, sample_block_size=sample_block_size)

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
            # #TODO
            # self.print_cnt+=1

            cov_param = cov_param_pred.clone()
            cov_param[:,1:2] = cov_param[:,0:1]+cov_param_pred[:,1:2]#!!
            cov_param[:,2:3] = cov_param[:,1:2]+cov_param_pred[:,2:3]#!!
            cov_param[:,3:]  = cov_param[:,3:].clone() / (torch.norm(cov_param_pred[:,3:], dim=-1, keepdim=True)+1e-9)
            # if self.print_cnt%1000==0:
            #     print("cov_param:",cov_param[len(cov_param)//2])

            eigval=torch.zeros(cov_param.shape[0],9, device=cov_param.device, dtype=cov_param.dtype)
            eigval[:,::4] = cov_param[:,:3]
            eigval=eigval.reshape(-1,3,3) 
            # eigvec=kornia.angle_axis_to_rotation_matrix(cov_param[:,3:] ) #Nx3x3
            eigvec=kornia.quaternion_to_rotation_matrix(cov_param[:,3:] ) #Nx3x3
            if not return_eig_vec:
                return eigvec@eigval@eigvec.transpose(-1,-2)
            else:
                return eigvec@eigval@eigvec.transpose(-1,-2), eigvec
        # xyz_pred = xyz_pred.to(dtype=torch.float32)
        # xyz_target=xyz_target.to(dtype=torch.float32)
        # normal_pred = normal_pred.to(dtype=torch.float32)
        # normal_target = normal_target.to(dtype=torch.float32)

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
            # weight_b = weight*weight
            # m,_ = torch.kthvalue(diff.reshape(-1), 1+int(len(diff.reshape(-1)-1 )*self.penalize_ratio), dim=-1)
            # m = torch.max(m, torch.ones_like(m))
            # count_mask = diff<m
            count_mask=self._points_roi(diff[None], penalize_ratio=self.penalize_ratio)

            # mask[b] = mask[b].squeeze(-1)[diff.squeeze(0)<m]
            mask_b = mask_b.squeeze(-1)[count_mask.squeeze(0)]
            mask_assoc_sel = mask_assoc.squeeze(-1)[count_mask.squeeze(0)]
            cov_pred[b] = cov_pred[b][count_mask.squeeze(0)]
            cov_pred_assoc = cov_pred_assoc[count_mask.squeeze(0)]
            diff_vec=diff_vec[count_mask.squeeze(0)]


           


            # rot_loss = 1-nn.functional.cosine_similarity(xyz_pred[b][None,...], xyz_assoc[None], dim=-1)#[...,None]

            # range_loss=(torch.norm(xyz_pred[b][None,...], dim=-1, keepdim=True)-torch.norm(xyz_assoc[None,...], dim=-1, keepdim=True))**2
            # sph_diff_ = rot_loss.squeeze(0)[diff.squeeze(0)<m]#+range_loss.squeeze(0)
            # diff_=range_diff_ = range_loss.squeeze(0)[diff.squeeze(0)<m].squeeze(-1) #N,
            if 0:#self.norm:
                square_diff =  diff_ * weight_b#mask_b#mask[b]  
                # square_diff_dyn =  diff_.detach() * mask_b#mask[b]  
                square_diff_dyn =  diff_ * mask_b*mask_assoc_sel #mask[b]  
                # square_sph_diff = diff_*mask_b.detach()*mask_assoc.detach()
            else:
                # square_diff = (diff_ * diff_)# * weight_b#mask_b #mask[b]  
                # print(diff_vec.shape, cov_pred[b].shape, R_pred.shape,cov_pred_assoc.shape, flush=True )
                sigma=cov_pred[b]+R_pred[b].detach()@cov_pred_assoc@R_pred[b].detach().transpose(-1,-2)
                try:
                    sigma_inv = torch.inverse(sigma)
                except:
                    # print(sigma.shape )
                    try:
                        sigma_inv=torch.inverse(sigma+torch.eye(3,device=sigma.device, dtype=sigma.dtype)*1e-6)
                    except:
                        print("Set segma_inv=eye")
                        sigma_inv=torch.eye(3,device=sigma.device, dtype=sigma.dtype)

                square_diff = (diff_vec.unsqueeze(-2)@sigma_inv @ diff_vec[...,None]).squeeze(-1) #N,   

                # square_diff_dyn = (diff_ * diff_).detach() * mask_b#mask_b #mask[b]  
                # square_diff_dyn = (diff_ * diff_) * mask_b*mask_assoc_sel#mask_b #mask[b]  
                # square_sph_diff = sph_diff_*mask_b.detach()*mask_assoc_sel.detach()

            input_shape = square_diff.shape
            # loss_ = torch.sum(square_diff, dim=list(range(1, len(input_shape))))/(torch.sum(weight_b, dim=list(range(1, len(input_shape)))) + 1e-12) 
            loss_ = torch.mean(square_diff)+self.reg_weight*torch.mean(0.5*torch.log(torch.det(sigma)))
            # loss_dyn_ = torch.mean(square_diff_dyn) \
            #     + self.reg_weight*torch.mean(torch.log(2/( mask[0][b].squeeze(-1)[diff.squeeze(0)<m]+1e-9) )) \
            #     + self.reg_weight*torch.mean(torch.log(2/(mask_assoc.squeeze(-1)[diff.squeeze(0)<m]+1e-9) )) \
            # loss_sph_ = torch.mean(square_sph_diff, )#/(torch.sum(mask_b.detach(), dim=list(range(1, len(input_shape)))) + 1e-12)
            # loss += loss_dyn_+self.sph_weight*loss_sph_
            loss += loss_

            #svd 
            src_mask = count_mask[...,None].expand_as(xyz_pred[b][None,...])
            src = xyz_pred[b][None,...][ src_mask ].reshape(1,-1,3).permute(0,2,1).detach()
            tgt = xyz_assoc[None,...][src_mask].reshape(1,-1,3).permute(0,2,1).detach()
            
            #TODO
            mask_b2=mask[0][b][None].squeeze(-1)[count_mask].reshape(1,-1).detach()
            mask_assoc_sel2=mask_assoc[None].squeeze(-1)[count_mask].reshape(1,-1).detach()
            wgt = weight[None].squeeze(-1)[count_mask].reshape(1,-1).detach() #
            # wgt = weight[None].squeeze(-1)[count_mask].reshape(1,-1).detach()* (1+mask_b2/mask_b2.max()*mask_assoc_sel2/mask_assoc_sel2.max() )# *mask[b][None].squeeze(-1)[count_mask].reshape(1,-1).detach()

            # try:
            #     R,t = self.svd(
            #         src,
            #         tgt,
            #         weight=wgt**2 )
            # except:
            #     print("svd failed", flush=True)
            #     R = torch.eye(3, device=wgt.device, dtype=src.dtype)[None]
            #     t = torch.zeros([1,3], device=weight.device, dtype=src.dtype)
            # res_R.append(R)
            # res_T.append(t)
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
                    # print(normal_pred[b].shape, orig_xyz_assoc.shape, xyz_pred[b].shape)
                    wgt =nn.functional.cosine_similarity(normal_pred[b], orig_xyz_assoc-xyz_pred[b], dim=-1)[...,None].abs() #Nx1

                    #compute roi_mask
                    roi_mask = self._points_roi(orig_diff[None], penalize_ratio=self.penalize_ratio) #1xN

                    src_mask = roi_mask[...,None].expand_as(xyz_pred[b][None,...]) #Nx1->1xNx3
                    src = xyz_pred[b][None,...][ src_mask ].reshape(1,-1,3).permute(0,2,1).detach()
                    tgt = orig_xyz_assoc[None,...][src_mask].reshape(1,-1,3).permute(0,2,1).detach()
                    wgt = wgt[None].squeeze(-1)[roi_mask].reshape(1,-1).detach()

                   

                # #Bx3x3 = Bx3x3@Bx3x3
                # res_r_ = R@res_r_
                # #Bx3 = (Bx3x3@Bx3x1+Bx3x1).squeeze(-1)
                # res_t_ = (R@res_t_[...,None]+t[...,None]).squeeze(-1)                
                     


            # res_R.append(R)
            # res_T.append(t)
            res_R.append(res_r_)
            res_T.append(res_t_)

        res_R=torch.cat(res_R, dim=0)
        res_T=torch.cat(res_T, dim=0)

        loss/=len(xyz_pred)
        # print(loss,flush=True)
            

        # focal_weight=1
        focal_weight = (torch.exp(-_alpha) * loss)**focal_gamma
        # focal_weight = focal_weight.detach()
        focal_weight = focal_weight/(torch.sum(focal_weight) + 1e-12)
        # focal_weight = focal_weight.detach()

        loss = focal_weight*(torch.exp(-_alpha) * loss)
        # loss =loss.mean()
        loss = loss.sum() + _alpha
        return loss, res_R, res_T


