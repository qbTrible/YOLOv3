import numpy as np
import torch


def Iou(box, boxes, isMin = False):
    box_area = box[4] * box[5]
    area = boxes[:, 4] * boxes[:, 5]
    xx1 = np.maximum(box[2]-box[4]/2, boxes[:, 2]-boxes[:, 4]/2)
    yy1 = np.maximum(box[3]-box[5]/2, boxes[:, 3]-boxes[:, 5]/2)
    xx2 = np.minimum(box[2]+box[4]/2, boxes[:, 2]+boxes[:, 4]/2)
    yy2 = np.minimum(box[3]+box[5]/2, boxes[:, 3]+boxes[:, 5]/2)

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))
    return ovr

def nms(boxes, thresh=0.3, isMin = False):

    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:, 1]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)

        # print(iou(a_box, b_boxes))
        # ovr = Iou(a_box, b_boxes, isMin)

        index = np.where(Iou(a_box, b_boxes, isMin) < thresh)
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)


# def nms(dets, iou_thr=0.3, method='gaussian', sigma=0.5, score_thr=0.01, isMin=False):
#     if not dets.any():
#         return np.array([])
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     areas = (x2 - x1) * (y2 - y1)
#     dets = np.concatenate((dets, areas[:, None]), axis=1)
#     retained_box = []
#     while dets.size > 0:
#         max_idx = np.argmax(dets[:, 4], axis=0)
#         dets[[0, max_idx], :] = dets[[max_idx, 0], :]
#         retained_box.append(dets[0, :-1])
#         iou = Iou(dets[0], dets[1:], isMin)
#         if method == 'linear':
#             weight = np.ones_like(iou)
#             weight[iou > iou_thr] -= iou[iou > iou_thr]
#         elif method == 'gaussian':
#             weight = np.exp(-(iou * iou) / sigma)
#         else:  # traditional nms
#             weight = np.ones_like(iou)
#             weight[iou > iou_thr] = 0
#         dets[1:, 4] *= weight
#         retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
#         dets = dets[retained_idx + 1, :]
#     return np.vstack(retained_box)


def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


if __name__ == '__main__':
    # a = np.array([1,1,11,11])
    # bs = np.array([[1,1,10,10],[11,11,20,20]])
    # print(iou(a,bs))

    bs = np.array([[1, 1, 10, 10, 40], [1, 1, 9, 9, 10], [9, 8, 13, 20, 15], [6, 11, 18, 17, 13]])
    # print(bs[:,3].argsort())
    print(nms(bs))
