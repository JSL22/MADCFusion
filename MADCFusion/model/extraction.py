
import torch.nn as nn
import torch
import numpy as np

# extraction network,    dilate_rate = [1, 3, 5],  channels =  [16, 32, 64]
class ExtractionNet(nn.Module):
    def __init__(self,):

        super(ExtractionNet, self,).__init__()
        channels = [16, 32, 64]
        self.extract_feature1 = HybridDilatedNet(in_channels=1, out_channels=channels[0], extension=0)
        self.extract_feature2 = HybridDilatedNet(in_channels=channels[0], out_channels=channels[1], extension=2)
        self.extract_feature3 = HybridDilatedNet(in_channels=channels[1], out_channels=channels[2], extension=4)

    def forward(self, in_x):
        in_x = in_x[:, :1]
        ef1 = self.extract_feature1(in_x)
        ef2 = self.extract_feature2(ef1)
        ef3 = self.extract_feature3(ef2)

        return ef1, ef2, ef3

#  Hybrid Dilated Convolution
class HybridDilatedNet(nn.Module):
    def __init__(self, in_channels=None, out_channels=None,
                 kernel_size=3, padding=None, stride=1, extension=None):
        super(HybridDilatedNet, self).__init__()
        #  123--125--126--127--133--134--135--136--137--138
        dilate_rate = [1, 3, 5]        # [1, 3, 5] | [1, 2 ,3]
        # dilate_rate = [1, 2, 3]        # [1, 3, 5] | [1, 2 ,3]
        base = 4
        out_channel = base * (2 * extension + 1)
        in_channel = base * (2 * extension + 2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel,
                               kernel_size=kernel_size, padding=dilate_rate[0], stride=stride, dilation=dilate_rate[0])
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=in_channel,
                               kernel_size=kernel_size, padding=dilate_rate[1], stride=stride, dilation=dilate_rate[1])
        self.bn2 = nn.BatchNorm2d(in_channel)

        # self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
        #                        kernel_size=1, padding=0, stride=stride, dilation=1)

        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channels,
                               kernel_size=kernel_size, padding=dilate_rate[2], stride=stride, dilation=dilate_rate[2])

        self.bn3 = nn.BatchNorm2d(out_channels)

        self.LRelu = nn.LeakyReLU()

    def forward(self, in_x):
        x1 = self.conv1(in_x)
        x1 = self.bn1(x1)
        x1 = self.LRelu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.LRelu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.LRelu(x3)

        return x3

if __name__ == "__main__":
    x = torch.tensor(np.random.rand(1, 3, 480, 640).astype(np.float32))
    extract_net = ExtractionNet()
    a, b, c = extract_net(x)
    print(a.shape)
    print(b.shape)
    print(c.shape)
