import torch
from torch import nn
import torch.nn.functional as F

from neuralop.models import TFNO, FNO

class Deconv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(
            1, 64, 
            kernel_size=(16, 1), 
            stride=(1, 1), 
            padding=(2, 26)
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 128, 
            kernel_size=(16, 1), 
            stride=(3, 1), 
            padding=(3, 26)
        )
        self.deconv3 = nn.ConvTranspose2d(
            128, 256, 
            kernel_size=(16, 1), 
            stride=(4, 1), 
            padding=(12, 26)
        )

    def forward(self, x, **kwargs):
        x = F.gelu(self.deconv1(x))
        x = F.gelu(self.deconv2(x))
        x = F.gelu(self.deconv3(x))

        return x
    
class NO1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(256, 160*100)
        self.fno = TFNO(
            n_modes=(16, 16),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            projection_channels=64, 
            factorization='tucker', 
            rank=0.42,
            # implementation='reconstructed',
        )

    def forward(self, x, **kwargs):
        x = F.gelu(F.dropout(self.lin(x)))
    
        x = x.view(-1, 1, 160, 100)
        x = self.fno(x)

        return x

class NO2(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = Deconv1()
        
        self.fno = TFNO(
            n_modes=(16, 16),
            in_channels=256,
            out_channels=1,
            hidden_channels=256,
            projection_channels=256, 
            factorization='tucker', 
            rank=0.42,
            # implementation='reconstructed',
        )

    def forward(self, x, **kwargs):
        x = self.deconv(x)
        x = self.fno(x)

        return x

class NO3(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.fno = TFNO(**config)

    def forward(self, x, **kwargs):
        # empty = torch.full_like(kwargs['y'], -1.)
        # empty[:, :, 160:192, :] = x.repeat(1, 1, 32, 1)
        
        x = self.fno(x)

        return x
    
    
class NO4(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.fno = TFNO(**config)

    def forward(self, x, **kwargs):
        empty = torch.full_like(kwargs['y'], -1.)
        empty[:, :, 40:48, :] = x.repeat(1, 1, 8, 1)
        
        x = self.fno(empty)

        return x
    
class CarstenNO(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.fno = TFNO(**config)
        
    def forward(self, **kwargs):
        # empty = torch.full_like(kwargs['y'], -1.)
        # empty[:, :, 160:192, :] = x.repeat(1, 1, 32, 1)
        
        x = self.fno(kwargs['x'])

        return x


# https://github.com/milesial/Pytorch-UNet/tree/master/unet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 2))

    def forward(self, x, **kwargs):
        # empty = torch.full_like(kwargs['y'], -1.)
        # empty[:, :, 160:192, :] = x.repeat(1, 1, 32, 1)
        # x = empty
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits


class UNet1(nn.Module):
    def __init__(self, n_channels=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.in_lin = nn.Linear(256, 160*100)

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        self.outc = OutConv(64, 1)

    def forward(self, x, **kwargs):
        x = F.relu(F.dropout(self.in_lin(x)))
        x = x.view(-1, 1, 160, 100)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)

        return x
    

class UNet2(nn.Module):
    def __init__(self, n_channels=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.deconv = Deconv1()

        self.inc = (DoubleConv(256, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        self.outc = OutConv(64, 1)

    def forward(self, x, **kwargs):
        x = self.deconv(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)

        return x