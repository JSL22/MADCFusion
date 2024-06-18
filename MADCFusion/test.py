import time
import torch
import os
import numpy as np
from model.reconstruction import FusionNet
from PIL import Image
from pre_dataset.loader import PrepareDataset
from torch.utils.data import DataLoader
from utils.utils import create_dir, parse_args, is_need_device

def test():
    batch_size = 1
    args = parse_args()
    num_workers = min([os.cpu_count(), batch_size if batch_size >= 1 else 0, 4])
    checkpoint = 'weight/fusion/fusion_model-80.pth'
    model = FusionNet()
    device = is_need_device()
    model.to(device)
    model.load_state_dict(torch.load(checkpoint,  map_location=device))
    model.eval()
    dataset = 'TNO'
    ir_path = './datasets/test/' + dataset +  '/ir/'
    vis_path = './datasets/test/' + dataset + '/vis/'
    fusion_dir = './fusion_result/' + dataset
    create_dir(fusion_dir)
    test_dataset = PrepareDataset(ir_path, vis_path, is_train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=False)
    start_time = time.perf_counter()
    with torch.no_grad():
        for index, (ir_img, vis_img, name) in enumerate(test_loader):
            ir_img = ir_img.to(device)
            vis_img = vis_img.to(device)
            if vis_img.shape[1] > 1:
                yuv_img = RGB2YCrCb(vis_img)
                fusion_img = model(ir_img, yuv_img)
                fusion_yuv =  torch.cat(
                   (fusion_img, yuv_img[:, 1:2, :, :],
                   yuv_img[:, 2:, :, :]),
                   dim=1,
                )
                fusion_img = YCrCb2RGB(fusion_yuv)
            else:
                fusion_img = model(ir_img, vis_img)

            ones = torch.ones_like(fusion_img)
            zeros = torch.zeros_like(fusion_img)
            fusion_img = torch.where(fusion_img > ones, ones, fusion_img)
            fusion_img = torch.where(fusion_img < zeros, zeros, fusion_img)

            fusion_img = fusion_img.cpu().numpy()
            fusion_img = fusion_img.transpose((0, 2, 3, 1))
            fusion_img = ((fusion_img - np.min(fusion_img))/ (np.max(fusion_img) - np.min(fusion_img)))
            fusion_img = np.uint8(255.0 * fusion_img)

            for i in range(len(name)):
                image = fusion_img[i, :, :, :]
                if image.shape[2] < 3:
                    image = image[:, :, 0]
                image = Image.fromarray(image)
                saved_fusion_img = os.path.join(fusion_dir, name[i])
                image.save(saved_fusion_img)
    end_time = time.perf_counter()
    print(f'run time : {end_time - start_time}')
def RGB2YCrCb(input_im):
    device = is_need_device()
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    device = is_need_device()
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

if __name__ == '__main__':
    print("=====start test=====")
    test()
    print("=====save all fused image=====")







