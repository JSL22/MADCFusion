import glob
import os
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

class PrepareDataset(Dataset):
    def __init__(self, ir_path=None, vis_path=None, transforms=None, is_train=None):
        super(PrepareDataset, self).__init__()
        self.is_train = is_train
        self.ir_names, self.ir_datum = get_image(ir_path)
        self.vis_names, self.vis_datum = get_image(vis_path)
        # assert (len(self.ir_name) == len(self.vis_name))
        self.length = min(len(self.ir_names), len(self.vis_names))
        self.transforms = transforms

    def __getitem__(self, item):
        img_name = self.ir_names[item]

        ir_image = self.ir_datum[item]
        ir_image = cv2.imread(ir_image, 0)
        ir_image = np.asarray(Image.fromarray(ir_image), dtype=np.float32) / 255.0
        ir_image = np.expand_dims(ir_image, axis=0)

        vis_image = self.vis_datum[item]
        if self.is_train:
            vis_image = np.array(Image.open(vis_image))
            vis_image = (np.asarray(Image.fromarray(vis_image), dtype=np.float32).transpose((2, 0, 1)) / 255.0)

        if self.is_train is False:
            vis_image = Image.open(vis_image)
            if len(vis_image.split()) > 1:
                vis_image = np.array(vis_image)
                vis_image = np.asarray(Image.fromarray(vis_image), dtype=np.float32).transpose((2, 0, 1)) / 255.0
            else:
                vis_image = np.array(vis_image, dtype=np.float32) / 255.0
                vis_image = np.expand_dims(vis_image, axis=0)
        if self.transforms is not None:
            pass
        return torch.tensor(ir_image), torch.tensor(vis_image), img_name

    def __len__(self):
        return self.length

def get_image(path):
    image_names = os.listdir(path)
    image_datum = []
    suffix = ['jpg', 'bmp', 'png']
    for suf in suffix:
        image_data = glob.glob(os.path.join(path, '*.' + suf))
        image_datum.extend(image_data)
    image_names.sort(key=lambda x: int(x[0:-4]))
    image_datum.sort(key=lambda x: int(x[len(path):-4]))
    return image_names, image_datum

if __name__ == '__main__':
    ir_path = '../datasets/train/Ir/'
    vis_path = '../datasets/train/Vis/'
    train_dataset = PrepareDataset(ir_path, vis_path, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True,
                                   num_workers=0, pin_memory=True)
    print(len(train_loader))
    for i, a in enumerate(train_loader):
        ir, vis, name = a
        if i == 3:
            print(vis.shape)