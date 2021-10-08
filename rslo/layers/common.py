import torch
import torch.nn as nn
import numpy as np

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class ExpandDims(nn.Module):
    def __init__(self, insert_dims, dim):
        super().__init__()
        self.insert_dims = insert_dims
        self.dim = dim

    def _new_shape(self, old_shape):
        new_shape = list(old_shape)
        del new_shape[self.dim]
        new_shape[self.dim:self.dim] = self.insert_dims
        return new_shape
    def forward(self, x):
        new_shape = self._new_shape(x.shape)
        return x.reshape(new_shape)

class ELUPlus(nn.ELU):
    def forward(self, x):
        return super(ELUPlus, self).forward(x)+1+1e-6


class EXP(nn.Module):
    def __init__(self, truncated_threshold=1e20):
        super().__init__()
        self.truncated_threshold = truncated_threshold
        self.truncated_point = np.log(self.truncated_threshold)

    def forward(self, x):
        x = torch.tanh(x/self.truncated_point)*self.truncated_point
        out = torch.exp(x)+1e-6
        return out


class ParameterLayer(nn.Module):
    def __init__(self, init_value, requires_grad=True):
        super(ParameterLayer, self).__init__()

        self.param = nn.Parameter(init_value, requires_grad=requires_grad)

    def forward(self, *input):
        return self.param


class MaskPropagator(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, max_pool_mask=True, groups=1):
        super(MaskPropagator, self).__init__()
        # assert(kernel_size % 2 == 1)
        self.max_pool_mask = max_pool_mask
        if max_pool_mask:
            self.mask_pool = nn.MaxPool2d(
                kernel_size, stride=stride, padding=padding)
            self.normalize_const = 1
        else:
            self.mask_pool = nn.Conv2d(1, 1,
                                       kernel_size, stride, bias=False, padding=padding)
            set_requires_grad(self.mask_pool, requires_grad=False)
            nn.init.constant_(self.mask_pool.weight, 1)
            self.normalize_const = 1
    def _conv(self, binary_mask):

        binary_mask = binary_mask
        mask = self.mask_pool(binary_mask) #/self.normalize_const
        if not self.max_pool_mask:
            # self.normalize_const = torch.max(mask, dim=(2,3), keepdim=True)
            # self.normalize_const,_ = torch.max(mask.view(mask.shape[0], -1), dim=-1, keepdim=True)
            # self.normalize_const = self.normalize_const.view(mask.shape[0], 1,1,1)
            self.normalizer_const = self.mask_pool((binary_mask>0).float() )
  
        mask /= self.normalize_const
        return mask.detach_()

    def forward(self, mask):
        mask = self._conv(mask)
        return mask 

class Dropout2dGivenMask(nn.Module):
    def __init__(self, p, dim=1):
        super(Dropout2dGivenMask, self).__init__()

        self.p = p
        self.dim = 1

    def forward(self, input, mask=None):
        if mask is None:
            p = torch.ones(input.shape[self.dim],
                           device=input.device) * (1-self.p)
            mask = torch.bernoulli(p).view(1, input.shape[self.dim], 1, 1)

        mask = mask.expand_as(input)
        input = torch.where(mask > 0, input, torch.zeros_like(input))

        return input, mask

class PCMaskGenerator(nn.Module):
    def __init__(self, kernel_sizes, strides, paddings, max_pool_mask=True, apply_func=None):
        super().__init__()
        assert isinstance(strides, (list, tuple))
        layer_num =  len(strides)  
        
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes]*layer_num
        if not isinstance(paddings, (list, tuple)):
            paddings = [paddings]*layer_num
        layers = [] 
        for k,s,p in zip(kernel_sizes, strides, paddings) :
           layers.append(MaskPropagator(1,1,k,s,p, max_pool_mask=max_pool_mask))
        
        self.layers = nn.ModuleList(layers)
        self._apply_func = apply_func
    
    def forward(self, x):

        masks = []
        for i in range(len(self.layers)):
            x = self.layers[i](x) 
            if self._apply_func is not None:
                masks.append(self._apply_func(x))
                # x = self._apply_func(x)
            else:
                masks.append(x)

        return masks
