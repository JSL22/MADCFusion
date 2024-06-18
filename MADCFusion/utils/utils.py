import time

import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np


# 计算梯度
class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        kernel_x = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_y = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        # result = torch.sqrt((grad_y**2 + grad_x**2))
        result =  torch.abs(grad_x) + torch.abs(grad_y)
        return result

# 结构化范数

# 模型参数
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--data-path", default="image/")  # 数据路径
    parser.add_argument("--weight-path", default="weight/fusion")  # 模型参数路径
    parser.add_argument("--log-path", default="log/")  # 模型参数路径
    parser.add_argument("--epochs", default=80, type=int, metavar="N")  # metavar：参数值提示
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W")
    args = parser.parse_args()
    return args

# 目录创建
def create_dir(dir_path):
    assert isinstance(dir_path, str), "dir_path is not String"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True

# 设备检测
def is_need_device():
    return torch.device('cuda' if torch.cuda.is_available() else "cpu")

# YUV色彩空间
def RGB2YCrCb(vis_img):
    device = is_need_device()
    b, c, h, w = vis_img.size()
    rbg_img = vis_img.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (bhw,c)
    R = rbg_img[:, 0]
    G = rbg_img[:, 1]
    B = rbg_img[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (temp.reshape(b, h, w, c).transpose(1, 3).transpose(2, 3))
    return out

def YCrCb2RGB(yuv_img):
    device = is_need_device()
    b, c, h, w = yuv_img.size()
    yuv = yuv_img.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (yuv + bias).mm(mat).to(device)
    out = (temp.reshape(b, h, w, c).transpose(1, 3).transpose(2, 3))
    return out
# if __name__ == '__main__':
    # img_path = '../image/test/1-vis/0011.png'
    # image = cv2.imread(img_path)
    # img = np.asarray(Image.fromarray(image), dtype=np.float32).transpose((2, 0, 1))
    # img = torch.tensor(img)
    # # c, h, w = img.shape
    # img = img.unsqueeze(0)
    # b, c, h, w = img.size()
    # re = RGB2YCrCb(img)
    # fusion_image = YCrCb2RGB(re)

