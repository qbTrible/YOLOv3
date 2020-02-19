import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import cfg
import os

from PIL import Image
import math

LABEL_FILE_PATH = "data01/label.txt"
IMG_BASE_DIR = "data01"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b


class MyDataset(Dataset):

    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def make_squre(self, im, max_size=416):
        im = Image.open(im)
        x, y = im.size
        scale = max(x, y) / 416
        size = (int(x / scale), int(y / scale))
        dx, dy = round((max_size - size[0]) / 2), round((max_size - size[1]) / 2)
        new_im = Image.new("RGB", (max_size, max_size))
        resize_im = im.resize(size, 1)
        new_im.paste(resize_im, (dx, dy))
        return new_im, scale, dx, dy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}

        line = self.dataset[index]
        strs = line.split()
        # _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))
        _img_data, scale, dx, dy = self.make_squre(os.path.join(IMG_BASE_DIR, strs[0]))
        img_data = transforms(_img_data)

        # _boxes = np.array([float(x) for x in strs[1:]])
        _boxes = np.array(list(map(float, strs[1:])))
        boxes = np.split(_boxes, len(_boxes) // 5)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))

            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf((cx / scale + dx) * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf((cy / scale + dy) * feature_size / cfg.IMG_WIDTH)
                w /= scale
                h /= scale

                for i, anchor in enumerate(anchors):
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]/(scale**2)
                    p_w, p_h = w / (anchor[0]/scale), h / (anchor[1]/scale)
                    p_area = w * h
                    intersection = np.minimum((anchor[0]/scale), w) * np.minimum((anchor[0]/scale), h)
                    # iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                    iou = intersection / (p_area + anchor_area - intersection)
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))])

        return labels[13], labels[26], labels[52], img_data
