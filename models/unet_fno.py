import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX > 0 or diffY > 0:
             x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class HybridUNetFNO(nn.Module):
    def __init__(self, in_channels, width, modes, bilinear=True):
        super(HybridUNetFNO, self).__init__()
        self.in_channels = in_channels
        self.width = width
        self.modes = modes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        
        self.fno_conv = SpectralConv2d(width * 4, width * 4, modes, modes)
        self.fno_skip = nn.Conv2d(width * 4, width * 4, 1)

        self.att1 = AttentionBlock(F_g=width * 4, F_l=width * 2, F_int=width * 2)
        self.up1 = Up(width * 6, width * 2, bilinear)
        
        self.att2 = AttentionBlock(F_g=width * 2, F_l=width, F_int=width)
        self.up2 = Up(width * 3, width, bilinear)

        self.outc = nn.Conv2d(width, 5, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x_fno = self.fno_conv(x3)
        x_skip = self.fno_skip(x3)
        x3 = F.gelu(x_fno + x_skip)
        
        x_up1 = self.up1.up(x3)
        diffY = x2.size()[2] - x_up1.size()[2]
        diffX = x2.size()[3] - x_up1.size()[3]
        if diffX > 0 or diffY > 0:
            x_up1 = F.pad(x_up1, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x2_att = self.att1(g=x_up1, x=x2)
        x_cat1 = torch.cat([x2_att, x_up1], dim=1)
        x_dec1 = self.up1.conv(x_cat1)
        
        x_up2 = self.up2.up(x_dec1)
        diffY = x1.size()[2] - x_up2.size()[2]
        diffX = x1.size()[3] - x_up2.size()[3]
        if diffX > 0 or diffY > 0:
            x_up2 = F.pad(x_up2, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x1_att = self.att2(g=x_up2, x=x1)
        x_cat2 = torch.cat([x1_att, x_up2], dim=1)
        x_dec2 = self.up2.conv(x_cat2)
        
        logits = self.outc(x_dec2)
        
        return {
            "mask": logits[:, 0:1, :, :],
            "h_mean": F.softplus(logits[:, 1:2, :, :]),
            "h_logvar": logits[:, 2:3, :, :],
            "u": logits[:, 3:4, :, :],
            "v": logits[:, 4:5, :, :]
        }
