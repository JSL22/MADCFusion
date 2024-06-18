import torch.nn as nn
from model.attention import ResGradAttention, GCNet
from model.extraction import ExtractionNet

class ConvBnLRelu(nn.Module):
    def __init__(self, in_channels=None, out_channels=None,
                 kernel_size=3, padding=1, stride=1, dilation=1):
        super(ConvBnLRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.LRelu = nn.LeakyReLU()

    def forward(self, in_x):
        in_x = self.conv(in_x)
        in_x = self.bn(in_x)
        in_x = self.LRelu(in_x)
        return in_x

class Conv1X1BnLRelu(nn.Module):
    def __init__(self, in_channels=None, out_channels=None,
                 kernel_size=1, padding=0, stride=1, dilation=1):
        super(Conv1X1BnLRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.LRelu = nn.LeakyReLU()

    def forward(self, in_x):
        in_x = self.conv(in_x)
        in_x = self.bn(in_x)
        in_x = self.LRelu(in_x)
        return in_x

class ConvBnTanH(nn.Module):
    def __init__(self, in_channels=None, out_channels=None,
                 kernel_size=3, padding=1, stride=1, dilation=1):
        super(ConvBnTanH, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.TanH = nn.Tanh()

    def forward(self, in_x):
        in_x = self.conv(in_x)
        in_x = self.bn(in_x)
        out_x = self.TanH(in_x)/ 2.0 + 0.5
        return out_x

class FusionNet(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(FusionNet, self).__init__()
        channels = [32, 16]

        self.extraction = ExtractionNet()
        self.attention16 = GCNet(channels[1])
        self.attention32 = ResGradAttention(channels[0], sobel=False)
        self.attention64 = ResGradAttention(in_channels, sobel=False)

        self.decode1 = ConvBnLRelu(in_channels, channels[0])
        self.decode2 = ConvBnLRelu(channels[0], channels[1])
        self.decode3 = ConvBnLRelu(channels[1], out_channels)

    def forward(self, ir_x, vis_x):
        ir_ext16, ir_ext32, ir_ext64 = self.extraction(ir_x)
        ir_att16 = self.attention16(ir_ext16)
        ir_att32 = self.attention32(ir_ext32)
        ir_att64 = self.attention64(ir_ext64)

        vis_x = vis_x[:, :1]
        vis_ext16, vis_ext32, vis_ext64 = self.extraction(vis_x)
        vis_att16 = self.attention16(vis_ext16)
        vis_att32 = self.attention32(vis_ext32)
        vis_att64 = self.attention64(vis_ext64)

        out_x64 = ir_att64 + vis_att64
        out_x32 = ir_att32 + vis_att32
        out_x16 = ir_att16 +  vis_att16

        recons_x32 = self.decode1(out_x64)
        recons_x32 = recons_x32 + out_x32

        recons_x16 = self.decode2(recons_x32)
        recons_x16 = recons_x16 + out_x16

        out_x = self.decode3(recons_x16)
        return out_x


if __name__ == '__main__':
    import numpy as np
    from torchsummary import summary
    import torch
    ir = torch.tensor(np.random.rand(1, 1, 100, 100).astype(np.float32))
    vis = torch.tensor(np.random.rand(1, 1, 100, 100).astype(np.float32))
    model = FusionNet()
    summary(model, [(1, 360, 480), (1, 360, 480)], device="cpu")    #Total params: 150,327 ，Estimated Total Size (MB): 4948.23
    y = model(ir, vis)


