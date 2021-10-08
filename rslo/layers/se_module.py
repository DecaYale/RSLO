from torch import nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y.expand_as(x)

class SpatialAttentionLayerV2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionLayerV2, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(channel//2, channel, kernel_size=3, padding=2, dilation=2),
            # nn.Conv2d(channel//2, channel, kernel_size=3, stride=2,padding=1),
            # nn.Conv2d(channel, channel, kernel_size=1, padding=0),
            # nn.Upsample(size=),
            nn.Conv2d(channel, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y.expand_as(x)
class SpatialAttentionLayerV3(nn.Module):
    def __init__(self, channel, reduction=16, LayerNorm=nn.InstanceNorm2d):
        super(SpatialAttentionLayerV3, self).__init__()
        print(f"SpatialAttentionLayerV3: LayerNone={LayerNorm}")
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, 2*channel, kernel_size=3, stride=2, padding=1, ),
            LayerNorm(2*channel),
            nn.LeakyReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*channel, 2*channel, kernel_size=3, stride=1, padding=1, ),
            LayerNorm(2*channel),
            nn.LeakyReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*channel, channel, kernel_size=3, stride=1, padding=1 ),
            LayerNorm(channel),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*channel, 1, kernel_size=1, stride=1, padding=0, ),
            nn.Sigmoid(),
        ) 
     

    def forward(self, x):
        x1= self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.deconv1(x1)
        x1 = torch.cat([x1,x], dim=1)
        y = self.conv3(x1)

        # y = self.attention(x)
        return x * y.expand_as(x)