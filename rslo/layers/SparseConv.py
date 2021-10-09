# from utils import utils
from torch.autograd import Variable as Variable
import torch.nn as nn
import torch
import sys
import apex
from .normalization import SemiGlobalSyncBatchNorm, MaskSyncBatchNorm
sys.path.insert(0, '../')

EPS1 = 1e-1
EPS12 = 1e-12


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class SPC_MaskSyncBN2d(MaskSyncBatchNorm):
    def __init__(self, 
    num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False, fuse_relu=False, 
    noise_scale_std=0, noise_shift_std=0
    ):
        super(SPC_MaskSyncBN2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, process_group, channel_last )

        self.noise_scale_std=noise_scale_std#kwargs.pop('noise_scale_std', 0)
        self.noise_shift_std=noise_shift_std#kwargs.pop('noise_shift_std', 0)
        if self.noise_scale_std!=0 or self.noise_shift_std!=0:
            self.add_noise=True
        else:
            self.add_noise=False
        # print(f"BN add noise: {self.add_noise}")

    def forward(self,x):
        if isinstance(x, (tuple, list)):
            tensor, binary_mask = x
        else:
            # tensor = x
            tensor, binary_mask=x, (x.abs().sum(dim=1,keepdim=True)>0).float().detach()#torch.ones_like(x[:,:1]) 
        tensor = super(SPC_MaskSyncBN2d, self).forward([tensor, binary_mask])
        # tensor = super(SPC_MaskSyncBN2d, self).forward(tensor )
            
        if self.add_noise:
            _,C,_,_=tensor.shape
            scale = torch.normal(mean=torch.ones(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_scale_std)
            shift = torch.normal(mean=torch.zeros(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_shift_std)
            # print(scale.shape, shift.shape, flush=True)
            tensor = (tensor+shift[None,:,None,None])*scale[None,:,None,None]
        
        if isinstance(x, (tuple, list)):
            return [tensor*binary_mask, binary_mask]
        else:
            return tensor*binary_mask

        # return tensor

class SPC_SemiGlobalSyncBN2d(SemiGlobalSyncBatchNorm):
    def __init__(self, 
    num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False, fuse_relu=False, 
    noise_scale_std=0, noise_shift_std=0
    ):
        super(SPC_SemiGlobalSyncBN2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, process_group, channel_last )

        self.noise_scale_std=noise_scale_std#kwargs.pop('noise_scale_std', 0)
        self.noise_shift_std=noise_shift_std#kwargs.pop('noise_shift_std', 0)
        if self.noise_scale_std!=0 or self.noise_shift_std!=0:
            self.add_noise=True
        else:
            self.add_noise=False
        # print(f"BN add noise: {self.add_noise}")

    def forward(self,x):
        if isinstance(x, (tuple, list)):
            tensor, binary_mask = x
            tensor = super(SPC_SemiGlobalSyncBN2d, self).forward(tensor)
            
            if self.add_noise:
                _,C,_,_=tensor.shape
                scale = torch.normal(mean=torch.ones(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_scale_std)
                shift = torch.normal(mean=torch.zeros(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_shift_std)
                # print(scale.shape, shift.shape, flush=True)
                tensor = (tensor+shift[None,:,None,None])*scale[None,:,None,None]
            return [tensor, binary_mask]
        else:
            tensor = super(SPC_SemiGlobalSyncBN2d, self).forward(x)
            if self.add_noise:
                _,C,_,_=tensor.shape
                scale = torch.normal(mean=torch.ones(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_scale_std)
                shift = torch.normal(mean=torch.zeros(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_shift_std)
                tensor = (tensor+shift[None,:,None,None] )*scale[None,:,None,None]

            return tensor
class SPC_SyncBN2d(apex.parallel.SyncBatchNorm):
    # def __init__(self, *args, **kwargs):
    def __init__(self, 
    num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False, fuse_relu=False, 
    noise_scale_std=0, noise_shift_std=0
    ):
        super(SPC_SyncBN2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, process_group, channel_last )

        self.noise_scale_std=noise_scale_std#kwargs.pop('noise_scale_std', 0)
        self.noise_shift_std=noise_shift_std#kwargs.pop('noise_shift_std', 0)
        if self.noise_scale_std!=0 or self.noise_shift_std!=0:
            self.add_noise=True
        else:
            self.add_noise=False
        # print(f"BN add noise: {self.add_noise}")

    def forward(self,x):
        if isinstance(x, (tuple, list)):
            tensor, binary_mask = x
            tensor = super(SPC_SyncBN2d, self).forward(tensor)
            
            if self.add_noise:
                _,C,_,_=tensor.shape
                scale = torch.normal(mean=torch.ones(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_scale_std)
                shift = torch.normal(mean=torch.zeros(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_shift_std)
                # print(scale.shape, shift.shape, flush=True)
                tensor = (tensor+shift[None,:,None,None])*scale[None,:,None,None]
            return [tensor, binary_mask]
        else:
            tensor = super(SPC_SyncBN2d, self).forward(x)
            if self.add_noise:
                _,C,_,_=tensor.shape
                scale = torch.normal(mean=torch.ones(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_scale_std)
                shift = torch.normal(mean=torch.zeros(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_shift_std)
                tensor = (tensor+shift[None,:,None,None] )*scale[None,:,None,None]

            return tensor
            # return super(SPC_SyncBN2d, self).forward(x)

# class SPC_BN2d(nn.BatchNorm2d):
#     # def __init__(self, *args, **kwargs):
#     #     super(SPC_BN2d, self).__init__(*args, **kwargs)

#     def forward(self, x):
#         if isinstance(x, (tuple, list)):
#             tensor, binary_mask = x
#             tensor = super(SPC_BN2d, self).forward(tensor)
#             return [tensor, binary_mask]
#         else:
#             return super(SPC_BN2d, self).forward(x)
class SPC_BN2d(nn.BatchNorm2d):
    def __init__(self, 
        num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False, fuse_relu=False, 
        noise_scale_std=0, noise_shift_std=0
    ):
        super(SPC_BN2d, self).__init__(num_features, eps, momentum, affine,track_running_stats)

        self.noise_scale_std=noise_scale_std#kwargs.pop('noise_scale_std', 0)
        self.noise_shift_std=noise_shift_std#kwargs.pop('noise_shift_std', 0)
        if self.noise_scale_std!=0 or self.noise_shift_std!=0:
            self.add_noise=True
        else:
            self.add_noise=False
        # print(f"BN add noise: {self.add_noise}")

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            tensor, binary_mask = x
            tensor = super(SPC_BN2d, self).forward(tensor)
            if self.add_noise:
                _,C,_,_=tensor.shape
                scale = torch.normal(mean=torch.ones(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_scale_std)
                shift = torch.normal(mean=torch.zeros(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_shift_std)
                # print(scale.shape, shift.shape, flush=True)
                tensor = (tensor+shift[None,:,None,None])*scale[None,:,None,None]
            
            return [tensor, binary_mask]
        else:
            tensor = super(SPC_BN2d, self).forward(x)
            if self.add_noise:
                _,C,_,_=tensor.shape
                scale = torch.normal(mean=torch.ones(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_scale_std)
                shift = torch.normal(mean=torch.zeros(C, dtype=tensor.dtype,device=tensor.device), std=self.noise_shift_std)
                tensor = (tensor+shift[None,:,None,None] )*scale[None,:,None,None]
            return tensor

class SPC_IN2d(nn.InstanceNorm2d):
    # def __init__(self, *args, **kwargs):
    #     super(SPC_BN2d, self).__init__(*args, **kwargs)
    
    def forward(self, x):
        if isinstance(x, (tuple, list)):
            tensor, binary_mask = x
            tensor = super(SPC_IN2d, self).forward(tensor)
            return [tensor, binary_mask]
        else:
            return super(SPC_IN2d, self).forward(x)


class SPC_ReLU(nn.ReLU):
    # def __init__(self, *args, **kwargs):
    #     super(SPC_ReLU, self).__init__(*args, **kwargs)

    def forward(self, x):
        if isinstance(x, (tuple, list)):

            tensor, binary_mask = x
            tensor = super(SPC_ReLU, self).forward(tensor)
            return [tensor, binary_mask]
        else:
            return super(SPC_ReLU, self).forward(x)


class SPC_LeakyReLU(nn.LeakyReLU):
    # def __init__(self, *args, **kwargs):
    #     super(SPC_LeakyReLU, self).__init__(*args, **kwargs)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            tensor, binary_mask = x
            tensor = super(SPC_LeakyReLU, self).forward(tensor)
            return [tensor, binary_mask]
        else:
            return super(SPC_LeakyReLU, self).forward(x)


class SparseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, max_pool_mask=True,**kwargs):
        super(SparseConv, self).__init__()
        assert(kernel_size % 2 == 1)

        self.out_channels = out_channels
        self.use_bias = bias
        self.max_pool_mask = max_pool_mask

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride, bias=False, padding=padding)
        self.sum_conv = nn.Conv2d(1, 1,
                                     kernel_size, stride, bias=False, padding=padding)
        set_requires_grad(self.sum_conv, requires_grad=False)
        nn.init.constant_(self.sum_conv.weight, 1)

        
        if max_pool_mask:
            self.mask_pool = nn.MaxPool2d(
                kernel_size, stride=stride, padding=padding)
            self.normalize_const = 1
        else:
            self.mask_pool = nn.Conv2d(1, 1,
                                       kernel_size, stride, bias=False, padding=padding)
            set_requires_grad(self.mask_pool, requires_grad=False)
            nn.init.constant_(self.mask_pool.weight, 1)

        # self.max_pool = nn.MaxPool2d(
        #     kernel_size, stride=stride, padding=padding)

        if self.use_bias:
            # self.b = nn.Parameter(torch.ones(out_channels),
            #                       requires_grad=True).cuda()
            self.b = nn.ParameterList(
                [nn.Parameter(torch.zeros(out_channels, 1, 1), requires_grad=True)])

            # nn.init.constant_(self.b[0], 0)
        else:
            self.b = [0]

    # def sparse_conv(self, tensor, binary_mask):
    def sparse_conv(self, tensor, mask):

        b, c, h, w = tensor.shape

        if mask is None:
            mask = torch.ones([b, 1, h, w]).cuda()
        mask = mask.detach()
        features = tensor*mask  # tf.multiply(tensor, binary_mask)

        # features = tf.layers.conv2d(features, filters=filters, kernel_size=kernel_size, strides=(
        #     strides, strides), trainable=True, use_bias=False, padding="valid", kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale))
        features = self.conv1(features)

        # norm = tf.layers.conv2d(binary_mask, filters=filters, kernel_size=kernel_size, strides=(
        #     strides, strides), kernel_initializer=tf.ones_initializer(), trainable=False, use_bias=False, padding="valid")
        norm = self.sum_conv(mask)

        # norm = tf.where(tf.equal(norm,0),tf.zeros_like(norm),tf.reciprocal(norm))
        norm = torch.where(norm ==0, torch.zeros_like(
            norm), 1./(norm + EPS12))
        _, bias_size, _, _ = norm.shape

        feature = features * norm + self.b[0]

        if not self.max_pool_mask:
            self.normalize_const,_ = torch.max(mask.view(mask.shape[0], -1), dim=-1, keepdim=True)
            self.normalize_const = self.normalize_const.view(mask.shape[0], 1,1,1)
    
        mask = self.mask_pool(mask)/self.normalize_const

        return [feature, mask.detach()]
        # return torch.cat([feature, mask], dim=1)

    # def forward(self, tensor, binary_mask):
    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x, (torch.sum(x, dim=1, keepdim=True) != 0).float()]
        tensor, mask = x
        return self.sparse_conv(tensor, mask)
