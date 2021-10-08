import torch
import torch.nn as nn


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class MaskMaxPool2d(nn.MaxPool2d):
    def forward(self,x):
        if isinstance(x, (list, tuple)):
            return super(MaskMaxPool2d, self).forward(x[0]), x[1]
        else:
            return super(MaskMaxPool2d, self).forward(x)

class MaskConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, max_pool_mask=True, groups=1):
        super(MaskConv, self).__init__()
        # assert(kernel_size % 2 == 1)

        self.out_channels = out_channels
        self.use_bias = bias
        # pad = kernel_size//2
        # self.pad = nn.ZeroPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride, bias=False, padding=padding, groups=groups)
        self.max_pool_mask = max_pool_mask
        if max_pool_mask:
            self.mask_pool = nn.MaxPool2d(
                kernel_size, stride=stride, padding=padding)
            self.normalize_const = 1
            print("max_pool_mask")
        else:
            self.mask_pool = nn.Conv2d(1, 1,
                                       kernel_size, stride, bias=False, padding=padding)
            set_requires_grad(self.mask_pool, requires_grad=False)
            nn.init.constant_(self.mask_pool.weight, 1)
            print("conv_mask")
            # self.normalize_const = kernel_size*kernel_size
            # def sparse_conv(self, tensor, binary_mask):
    def _conv(self, tensor, binary_mask):

        # if binary_mask is None:
        #     binary_mask = torch.ones([b, 1, h, w]).cuda()
        # binary_mask = binary_mask.detach() ##comment on 14/11/19
        # print(binary_mask.requires_grad, '!!!')
        binary_mask = binary_mask
        tensor = self.conv1(tensor)
        mask = self.mask_pool(binary_mask) #/self.normalize_const
        if not self.max_pool_mask:
            # self.normalize_const = torch.max(mask, dim=(2,3), keepdim=True)
            self.normalize_const,_ = torch.max(mask.view(mask.shape[0], -1), dim=-1, keepdim=True)
            self.normalize_const = self.normalize_const.view(mask.shape[0], 1,1,1)
  
        mask /= self.normalize_const
        # print(mask.requires_grad, '!!!!')
        # return tensor, mask.detach() #comment on 14/11/19
        return tensor, mask.detach_()

    def forward(self, x):

        if not isinstance(x, (list, tuple)):
            x = [x, (torch.sum(x.abs(), dim=1, keepdim=True) != 0).float().detach()]
        tensor, binary_mask = x
       
        tensor, binary_mask = self._conv(tensor, binary_mask)
      
        return [tensor, binary_mask]


class MaskConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(MaskConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride=1,
                                                  padding=0, output_padding=0, groups=1, bias=True, dilation=1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x, (torch.sum(x, dim=1, keepdim=True) != 0).float().detach()]
        tensor, binary_mask = x
        return super(MaskConvTranspose2d, self).forward(tensor), binary_mask.detach()
