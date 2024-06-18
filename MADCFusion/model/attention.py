
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F

class ResGradAttention(nn.Module):
    def __init__(self, in_channels, sobel=False):
        super(ResGradAttention, self).__init__()
        self.sobel = sobel
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels//16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels//16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        if sobel:
            self.resdualGrad = nn.Sequential(
                SobelNet(in_channels)
                # self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                #                          kernel_size=1, stride=1, padding=0, dilation=1)
            )


    def forward(self, in_x):
        x1 = self.se(in_x)
        out_x1 = in_x*x1
        out_x = out_x1
        if self.sobel:
            in_x2 = self.resdualGrad(in_x)
            # out_x = out_x1 + in_x2
            x2 = self.se(in_x2)
            out_x2 = in_x2*x2
            out_x = out_x1 + out_x2
        return out_x

class SobelNet(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super(SobelNet, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convX = nn.Conv2d(in_channels, in_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.convX.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convY = nn.Conv2d(in_channels, in_channels,
                               kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.convY.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, in_x):
        sobelX = self.convX(in_x)
        sobelY = self.convY(in_x)
        out_x = torch.abs(sobelX) + torch.abs(sobelY)
        return out_x

'''
    类比GCNet
    空间attention
'''
class GCNet(nn.Module):
    def __init__(self, in_channels=None, sobel=False):
        super(GCNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=1,
                               kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 16,
                               kernel_size=1, stride=1, padding=0)
        self.LNorm = nn.LayerNorm([in_channels // 16, 1, 1])
        self.Relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=in_channels // 16, out_channels=in_channels,
                               kernel_size=1, stride=1, padding=0)
        self.sobel = sobel
        if sobel:
            self.resdualGrad = nn.Sequential(
                SobelNet(in_channels),
                # nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                #                                kernel_size=1, stride=1, padding=0, dilation=1),
            )

    def forward(self, in_x):
        bs, channel, _, _ = in_x.size()
        in_k = self.conv1(in_x)
        in_k = in_k.view(bs, 1, -1).permute(0, 2, 1)
        in_k = F.softmax(in_k, dim=1)

        in_x1 = in_x.view(bs, channel, -1)
        in_v1 = torch.matmul(in_x1, in_k)
        in_v1 = in_v1.view(bs, channel, 1, 1)

        in_v1 = self.conv2(in_v1)
        in_v1 = self.LNorm(in_v1)
        in_v1 = self.Relu(in_v1)
        in_v2 = self.conv3(in_v1)

        out_x1= in_x + in_v2
        out_x = out_x1

        if self.sobel:
            in_x2 = self.resdualGrad(in_x)
            # out_x = out_x1 + in_x2
            # x2 = self.se(in_x2)
            # out_x2 = in_x2*x2
            out_x = out_x1 + in_x2

        return out_x


if __name__ == '__main__':
    # from extraction import ExtractionNet
    # extract_net = ExtractionNet()
    # x = torch.tensor(np.random.rand(1, 3, 460, 630).astype(np.float32))
    # a, b, c = extract_net(x)
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    # channels = [32, 64, 128]
    # model = ResGradAttention(channels[0])
    # y = model(a)
    # print(y.shape)
    x = torch.rand(10, 16, 100, 100)
    model = GCNet(16)
    y = model(x)
    print(y.shape)

