import torch
from torch import nn
from dataloader import MyData
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        return out

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


### 降采样
class Downsample(nn.Module):
     def __init__(self, in_channels, out_channels):
         super(Downsample, self).__init__()
         self.conv_relu1 = nn.Sequential(
             nn.Conv2d(in_channels, out_channels,
                       kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
         )
         self.conv_relu2 = nn.Sequential(
             nn.Conv2d(out_channels, out_channels,
                       kernel_size=3, padding=1),
             nn.ReLU(inplace=True)
         )
         self.pool = nn.MaxPool2d(kernel_size=2)

     def forward(self, x, is_pool=True):
         if is_pool:
             x = self.pool(x)
         x = self.conv_relu1(x)
         x = self.conv_relu2(x)
         return x

##  上采样
class Upsample1(nn.Module):

      def __init__(self, channels):
         super(Upsample1, self).__init__()
         self.conv_relu = nn.Sequential(
             nn.Conv2d(2*channels, channels,
                       kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(channels, channels,
                       kernel_size=3, padding=1),
             nn.ReLU(inplace=True)
         )
         self.upconv_relu = nn.Sequential(
             nn.ConvTranspose2d(channels,
                                channels//2,
                                kernel_size=(4, 3),
                                stride=2,
                                padding=1,
                                dilation=1,
                                output_padding=1),
             nn.ReLU(inplace=True)
         )

      def forward(self, x):
          x = self.conv_relu(x)

          x = self.upconv_relu(x)

          return x

##  上采样
class Upsample2(nn.Module):

      def __init__(self, channels):
         super(Upsample2, self).__init__()
         self.conv_relu = nn.Sequential(
             nn.Conv2d(2*channels, channels,
                       kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(channels, channels,
                       kernel_size=3, padding=1),
             nn.ReLU(inplace=True)
         )
         self.upconv_relu = nn.Sequential(
             nn.ConvTranspose2d(channels,
                                channels//2,
                                kernel_size=(3, 4),
                                stride=2,
                                padding=1,
                                dilation=1,
                                output_padding=1),
             nn.ReLU(inplace=True)
         )

      def forward(self, x):
          x = self.conv_relu(x)
          x = self.upconv_relu(x)
          return x

##  上采样
class Upsample3(nn.Module):

      def __init__(self, channels):
         super(Upsample3, self).__init__()
         self.conv_relu = nn.Sequential(
             nn.Conv2d(2*channels, channels,
                       kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(channels, channels,
                       kernel_size=3, padding=1),
             nn.ReLU(inplace=True)
         )
         self.upconv_relu = nn.Sequential(
             nn.ConvTranspose2d(channels,
                                channels//2,
                                kernel_size=(4, 4),
                                stride=2,
                                padding=1,
                                dilation=1,
                                output_padding=1),
             nn.ReLU(inplace=True)
         )

      def forward(self, x):
          x = self.conv_relu(x)
          x = self.upconv_relu(x)
          return x

class  Unet_model(nn.Module):
    def __init__(self):
        super(Unet_model, self).__init__()
        self.down1 = Downsample(15, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 1024)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(1024,
                               512,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               dilation=2,
                               output_padding=0),
            nn.ReLU(inplace=True)
        )
        self.up1 = Upsample1(512)
        self.up2 = Upsample2(256)
        self.up3 = Upsample3(128)
        self.conv_2 = Downsample(128, 64)
        self.last = nn.Conv2d(64, 9, kernel_size=1)
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)

    def forward(self, x):
        x1 = self.down1(x, is_pool=False)     # B*64*61*299
        x1_1 = self.cbam1(x1) + x1
        x2 = self.down2(x1_1)                 # B*128*30*149
        x2_1 = self.cbam2(x2) + x2
        x3 = self.down3(x2_1)                 # B*256*15*74
        x3_1 = self.cbam3(x3) + x3

        x4 = self.down4(x3_1)                 # B*512*7*37
        x4_1 = self.cbam4(x4) + x4

        x5 = self.down5(x4_1)                 # B*1024*3*18
        x5 = self.up(x5)                    # B*512*7*37
        x5 = torch.cat([x4, x5], dim=1)     # B*1024*7*37
        x5 = self.up1(x5)                   # B*256*15*74
        x5 = torch.cat([x3, x5], dim=1)     # B*512*15*74
        x5 = self.up2(x5)                   # B*128*30*149
        x5 = torch.cat([x2, x5], dim=1)     # B*256*30*149
        x5 = self.up3(x5)                   # B*64*61*299
        x5 = torch.cat([x1, x5], dim=1)     # B*128*61*299
        x5 = self.conv_2(x5, is_pool=False) # B*64*61*299
        x5 = self.last(x5)
        return x5
