from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.utils import Gradient


# 内容损失
# 超参数 p， g, gama
class ContentLoss(nn.Module):
    def __init__(self, p, g, gama):
        super(ContentLoss, self).__init__()
        self.gradient = Gradient()
        self.p = p
        self.g = g
        self.gama = gama

    def forward(self, in_ir, in_vis,  fusion_img):
        in_ir = in_ir[:, :1, :, :]
        in_vis = in_vis[:, :1, :, :]
        fusion_img = fusion_img[:, :1, :, :]
        # 像素级
        in_pixel = self.p * in_ir + (1 - self.p) * in_vis
        loss_pixel = F.l1_loss(in_pixel, fusion_img)
        # 梯度级
        fusion_img_grad = self.gradient(fusion_img)
        ir_grad = self.gradient(in_ir)
        vis_grad = self.gradient(in_vis)
        in_grad = self.g * vis_grad + (1 - self.g) * ir_grad
        loss_grad = F.l1_loss(in_grad, fusion_img_grad)
        # 内容损失
        loss_content = loss_pixel + self.gama * loss_grad
        return loss_content, loss_pixel, loss_grad