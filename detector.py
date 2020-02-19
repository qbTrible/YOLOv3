from module import *
import cfg
from PIL import Image, ImageDraw, ImageFont
import torchvision
import os
import torch.nn as nn
from tool import utils


class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = Darknet53().cuda()
        if os.path.exists("model/darknet01.pth"):
            self.net.load_state_dict(torch.load("model/darknet01.pth"))

        self.net.eval()

    def forward(self, input, thresh, anchors):
        output_13, output_26, output_52 = self.net(input.cuda())

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        boxes = torch.cat([boxes_13, boxes_26, boxes_52], dim=0).cpu().detach().numpy()
        print(boxes.shape)
        return utils.nms(boxes, 0.5, isMin=False)

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        output[..., 0:3] = nn.Sigmoid()(output[..., 0:3])

        mask = output[..., 0] > thresh
        idxs = mask.nonzero()
        vecs = output[mask]

        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        if vecs.shape[0] == 0:
            return torch.Tensor([]).cuda()
        anchors = torch.Tensor(anchors).cuda()

        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x

        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])

        d = vecs[:, 0]
        cls = vecs[:, 5:].argmax(1).float()

        return torch.stack([n.float(), d, cx, cy, w, h, cls], dim=1)

def make_squre(im, max_size=416):
    im = Image.open(im)
    x, y = im.size
    scale = max(x, y) / 416
    size = (int(x / scale), int(y / scale))
    dx, dy = round((max_size - size[0]) / 2), round((max_size - size[1]) / 2)
    new_im = Image.new("RGB", (max_size, max_size))
    resize_im = im.resize(size, 1)
    new_im.paste(resize_im, (dx, dy))
    return im, new_im, scale, dx, dy


if __name__ == '__main__':

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    # img = Image.open("data/images/1.jpg")
    img, new_img, scale, dx, dy = make_squre("data01/images/6.jpg")
    data = transforms(new_img).unsqueeze(0)
    detector = Detector()
    y = detector(data.cuda(), 0.45, cfg.ANCHORS_GROUP)
    # print(y)
    if y.shape[0] != 0:
        y[:, 2] = (y[:, 2] - dx) * scale
        y[:, 3] = (y[:, 3] - dy) * scale
        # y[:, 4:6] *= scale
        print(y)
        boxes = y
        boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5] = y[:, 2] - y[:, 4]/2, y[:, 3] - y[:, 5]/2, y[:, 2] + y[:, 4]/2, y[:, 3] + y[:, 5]/2

        draw = ImageDraw.Draw(img)
        color = ["red", "blue", "yellow", "green", "pink"]
        labels = {0: "人", 1: "马", 2: "猫", 3: "狗", 4: "车"}
        font = ImageFont.truetype("SIMLI.TTF", 15)

        for box in boxes:
            draw.rectangle(box[2:6], outline=color[int(box[-1])], width=3)
            chars = labels[int(box[-1])]+": "+str(round(box[1], 3))
            chars_w, chars_h = font.getsize(chars)
            if box[3] - chars_h - 4 >= 0:
                draw.rectangle([box[2], box[3] - chars_h - 4, box[2] + chars_w + 4, box[3]], fill=color[int(box[-1])])
                draw.text([box[2]+3, box[3]-chars_h - 3], labels[int(box[-1])]+": "+str(round(box[1], 3)), "black", font=font)
            else:
                draw.rectangle([box[2], box[3], box[2] + chars_w + 4, box[3] + chars_h + 4], fill=color[int(box[-1])])
                draw.text([box[2] + 3, box[3] + 2], labels[int(box[-1])] + ": " + str(round(box[1], 3)), "black",
                          font=font)
        img.show()
