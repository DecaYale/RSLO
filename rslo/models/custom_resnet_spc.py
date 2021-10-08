

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from rslo.layers.SparseConv import SparseConv, SPC_BN2d, SPC_LeakyReLU, SPC_ReLU
# from rslo.layers.se_module import SELayer, SpatialAttentionLayer ,SpatialAttentionLayerV2,SpatialAttentionLayerV3
import torch

# Conv2d = SparseConv


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

EPS12 = 1e-12


def conv1x1(in_planes, out_planes, stride=1, Conv2d=None, groups=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                  padding=0, bias=False, groups=groups)

def conv3x3(in_planes, out_planes, stride=1, Conv2d=None, groups=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False, groups=groups)

# only add custom bn to BasicBlock and Bottleneck


def SPC_add(a, b):
    if isinstance(a, (list, tuple)):
        # return [(a[0]*a[1]+b[0]*b[1])/(a[1]+b[1]+EPS12), (a[1]+b[1] > 0).float()]
        # return [a[0]+b[0], (a[1]+b[1] > 0).float()]
        return [a[0]+b[0], ((a[1]+b[1])/2).float()]  # changed on 14/11/19
        # return [a[0]+b[0], torch.max(a[1],b[1]).float()]  # changed on 14/11/19
    else:
        return a+b

def SPC_cat(a, b):
    if isinstance(a, (list, tuple)):
        return [torch.cat( [a[0],b[0]], dim=1 ), ((a[1]+b[1])/2).float()]  
    else:
        return torch.cat([a,b], dim=1) 

# class SEBasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None,
#                  *, reduction=16):
#         super(SEBasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes, 1)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.se = SELayer(planes, reduction)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.se(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out
class FireBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, lrelu=False, BN=None, Conv2d=None, groups=1, use_se=False, use_sa=False):
        super(FireBlock, self).__init__()
        if BN is None:
            self.BatchNorm2d = nn.BatchNorm2d
        else:
            self.BatchNorm2d = BN

        if Conv2d is None:
            self.Conv2d = nn.Conv2d
        else:
            self.Conv2d = Conv2d

        self.conv1 = conv1x1(inplanes, planes, stride,
                             Conv2d=self.Conv2d, groups=groups)
        self.bn1 = self.BatchNorm2d(planes)
        # self.relu = nn.LeakyReLU(0.1, True) if lrelu else nn.ReLU(inplace=True)
        self.relu = SPC_LeakyReLU(
            0.1, True) if lrelu else SPC_ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes, stride, Conv2d=self.Conv2d, groups=groups)
        self.bn2 = self.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.use_sa = use_sa
        if use_se:
            self.se = SELayer(planes*2, 16)
        if use_sa:
            # self.sa = SpatialAttentionLayer(planes)
            # self.sa = SpatialAttentionLayerV2(planes)
            self.sa = SpatialAttentionLayerV3(planes*2)
    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        # out = torch.cat([out,out2], dim=1) 
        out = SPC_cat(out,out2) 
        if self.use_sa:
            if not isinstance(out, (list, tuple)):
                out = out+self.sa(out)
            else:
                out = [self.sa(out[0]), out[1]]

        if self.use_se:
            if not isinstance(out, (list, tuple)):
                out = self.se(out)
            else:
                out = [self.se(out[0]), out[1]]

        # if self.downsample is not None:
        #     residual = self.downsample(residual)

        # out += residual
        # out = SPC_add(out, residual)
        # out = self.relu(out)

        return out

class BasicBlockZero(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, lrelu=False, BN=None, Conv2d=None, groups=1, use_se=False, use_sa=False):
        super(BasicBlockZero, self).__init__()
        if BN is None:
            self.BatchNorm2d = nn.BatchNorm2d
        else:
            self.BatchNorm2d = BN

        if Conv2d is None:
            self.Conv2d = nn.Conv2d
        else:
            self.Conv2d = Conv2d

        self.conv1 = conv3x3(inplanes, planes, stride,
                             Conv2d=self.Conv2d, groups=groups)
        self.bn1 = self.BatchNorm2d(planes)
        # self.relu = nn.LeakyReLU(0.1, True) if lrelu else nn.ReLU(inplace=True)
        self.relu = SPC_LeakyReLU(
            0.1, True) if lrelu else SPC_ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, Conv2d=self.Conv2d, groups=groups)
        self.bn2 = self.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.use_sa = use_sa
        if use_se:
            self.se = SELayer(planes, 16)
        if use_sa:
            # self.sa = SpatialAttentionLayer(planes)
            # self.sa = SpatialAttentionLayerV2(planes)
            self.sa = SpatialAttentionLayerV3(planes,LayerNorm=self.BatchNorm2d)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out[0]*=out[1].detach()
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out[0]*=out[1].detach()


        if self.downsample is not None:
            residual = self.downsample(x)
            residual[0]*=residual[1].detach()

        # out += residual
        out = SPC_add(out, residual)
        out = self.relu(out)

        if self.use_sa:
            if not isinstance(out, (list, tuple)):
                out = out+self.sa(out)
            else:
                # out = [self.sa(out[0]), out[1]]
                out = [out[0]+self.sa(out[0]), out[1]]
        if self.use_se:
            if not isinstance(out, (list, tuple)):
                out = self.se(out)
            else:
                out = [self.se(out[0]), out[1]]

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, lrelu=False, BN=None, Conv2d=None, groups=1, use_se=False, use_sa=False):
        super(BasicBlock, self).__init__()
        if BN is None:
            self.BatchNorm2d = nn.BatchNorm2d
        else:
            self.BatchNorm2d = BN

        if Conv2d is None:
            self.Conv2d = nn.Conv2d
        else:
            self.Conv2d = Conv2d

        self.conv1 = conv3x3(inplanes, planes, stride,
                             Conv2d=self.Conv2d, groups=groups)
        self.bn1 = self.BatchNorm2d(planes)
        # self.relu = nn.LeakyReLU(0.1, True) if lrelu else nn.ReLU(inplace=True)
        self.relu = SPC_LeakyReLU(
            0.1, True) if lrelu else SPC_ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, Conv2d=self.Conv2d, groups=groups)
        self.bn2 = self.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.use_sa = use_sa
        if use_se:
            self.se = SELayer(planes, 16)
        if use_sa:
            # self.sa = SpatialAttentionLayer(planes)
            # self.sa = SpatialAttentionLayerV2(planes)
            self.sa = SpatialAttentionLayerV3(planes,LayerNorm=self.BatchNorm2d)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.use_sa:
        #     if not isinstance(out, (list, tuple)):
        #         out = out+self.sa(out)
        #     else:
        #         out = [self.sa(out[0]), out[1]]

        # if self.use_se:
        #     if not isinstance(out, (list, tuple)):
        #         out = self.se(out)
        #     else:
        #         out = [self.se(out[0]), out[1]]

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = SPC_add(out, residual)
        out = self.relu(out)

        if self.use_sa:
            if not isinstance(out, (list, tuple)):
                out = out+self.sa(out)
            else:
                # out = [self.sa(out[0]), out[1]]
                out = [out[0]+self.sa(out[0]), out[1]]
        if self.use_se:
            if not isinstance(out, (list, tuple)):
                out = self.se(out)
            else:
                out = [self.se(out[0]), out[1]]

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, lrelu=False, BN=None, Conv2d=None):
        super(Bottleneck, self).__init__()
        if BN is None:
            self.BatchNorm2d = nn.BatchNorm2d
        else:
            self.BatchNorm2d = BN
        if Conv2d is None:
            self.Conv2d = nn.Conv2d
        else:
            self.Conv2d = Conv2d
        self.conv1 = self.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.BatchNorm2d(planes)
        self.conv2 = self.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False)
        self.bn2 = self.BatchNorm2d(planes)
        self.conv3 = self.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.BatchNorm2d(planes * 4)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(0.1, True) if lrelu else nn.ReLU(inplace=True)
        self.relu = SPC_LeakyReLU(
            0.1, True) if lrelu else SPC_ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = SPC_add(out, residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, lrelu=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.lrelu = lrelu
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(
            0.1, True) if self.lrelu else nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, lrelu=self.lrelu))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, lrelu=self.lrelu))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Using resnet 18")
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet18_h(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
