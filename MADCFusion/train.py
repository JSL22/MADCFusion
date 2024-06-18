import datetime
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.reconstruction import FusionNet
from pre_dataset.loader import PrepareDataset
from utils.log import initialize_log
from utils.loss import ContentLoss
from utils.utils import parse_args, create_dir, is_need_device


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def train():
    # same_seeds(10)
    args = parse_args()
    batch_size = args.batch_size
    device = is_need_device()
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    model = FusionNet()
    model.to(device)

    ir_path = './datasets/train/Ir/'
    vis_path = './datasets/train/Vis/'
    train_dataset = PrepareDataset(ir_path, vis_path, is_train=True)
    print(f'======Start preparing data with a length of {train_dataset.length}======')

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    train_len = len(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    conLoss = ContentLoss(0.65, 0.92, 100)

    log_path = args.log_path
    initialize_log(log_path + '/data')
    logger = logging.getLogger()
    writer = SummaryWriter(log_dir=log_path + '/board')
    weight_path = args.weight_path
    create_dir(weight_path)

    start_time = one_start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        for index, (ir_img, vis_img, name) in enumerate(train_loader):
            ir_img = ir_img.to(device)
            vis_img = vis_img.to(device)
            yuv_img = RGB2YCrCb(vis_img)
            fusion_img = model(ir_img, yuv_img)
            optimizer.zero_grad()
            loss_content = conLoss(ir_img, yuv_img, fusion_img)
            loss_content.backward()

            # nn.utils.clip_grad_norm(model.parameters(), max_norm=10)
            optimizer.step()

            iter_count = epoch * train_len + index + 1
            end_time = time.time()
            one_iter_time = end_time - one_start_time
            iter_time = (end_time - start_time) / iter_count
            one_start_time = end_time
            consume_time = str(datetime.timedelta(seconds=iter_time))

            if (iter_count % 12) == 0:
                data = (
                    f'iter: {iter_count} / {train_len * args.epochs}',
                    f'loss_content : {loss_content.item()}',
                    f'one_iter_time: {one_iter_time}',
                    f'iter_time: {iter_time:.4f}',
                    f'consume_time: {consume_time}'
                )
                msg = '-----'.join(data)
                logger.info(msg)
            writer.add_scalar('loss_content', loss_content.item(), iter_count)

        fusion_checkpoint = os.path.join(weight_path, f'fusion_model-{epoch + 1}.pth')
        torch.save(model.state_dict(), fusion_checkpoint)
        logger.info("=== End of epoch {epoch}, Fusion model is saved to {model}".
                    format(epoch=epoch + 1, model=fusion_checkpoint))
    total_time = time.time() - start_time
    model_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
    logger.info("=======Total consumption time {total_time}=={model_time}=======".format(total_time=total_time, model_time=model_time))

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
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
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
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
    print('=====start train=====')
    train()
