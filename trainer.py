import dataset
from module import *
import cfg
import torch.nn as nn
import os
import torch.optim.lr_scheduler as lr_scheduler


def loss_fn(output, target, alpha):
    entropy = nn.CrossEntropyLoss() # 分类损失函数
    mse = nn.MSELoss() # w, h 均方差损失
    bce = nn.BCELoss() # cx, cy, iou损失函数

    # 变换数据格式
    output = output.permute(0, 2, 3, 1)
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
    output[..., 0:3] = nn.Sigmoid()(output[..., 0:3])
    target = target.float()

    # 分离有目标和无目标的索引
    mask_obj = target[..., 0] > 0
    mask_noobj = target[..., 0] == 0

    # 分类（只做有目标损失）
    output_cls = output[mask_obj][..., 5:].reshape(-1, cfg.CLASS_NUM)
    target_cls = target[mask_obj][..., 5:].reshape(-1, cfg.CLASS_NUM).argmax(1)
    loss_cls = entropy(output_cls, target_cls) # 分类误差

    # cx, cy, w, h（只做有目标损失）
    loss_cxy = bce(output[mask_obj][..., 1:3], target[mask_obj][..., 1:3]) # cx, cy损失
    loss_cwh = mse(output[mask_obj][..., 3:5], target[mask_obj][..., 3:5]) # w, h损失
    loss_offset = loss_cxy + loss_cwh

    # iou（有目标和无目标权重不同）
    loss_iou_obj = bce(output[mask_obj][..., 0:1], target[mask_obj][..., 0:1]) # 有目标损失
    loss_iou_noobj = bce(output[mask_noobj][..., 0:1], target[mask_noobj][..., 0:1]) # 无目标损失
    loss_iou = 0.7*loss_iou_obj + 0.3*loss_iou_noobj # iou总损失

    return loss_offset, loss_cls, loss_iou


if __name__ == '__main__':

    myDataset = dataset.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=2, shuffle=True)

    net = Darknet53().cuda()
    if os.path.exists("model/darknet.pth"):
        net.load_state_dict(torch.load("model/darknet.pth"))
    net.train()

    # opt = torch.optim.Adam(net.parameters())
    opt = torch.optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
    scheduler = lr_scheduler.StepLR(opt, 20, gamma=0.8)
    epochs = 0
    while True:
        scheduler.step()
        for i, (target_13, target_26, target_52, img_data) in enumerate(train_loader):
            output_13, output_26, output_52 = net(img_data.cuda())
            loss_offset_13, loss_cls_13, loss_iou_13 = loss_fn(output_13, target_13.cuda(), 0.9)
            loss_offset_26, loss_cls_26, loss_iou_26 = loss_fn(output_26, target_26.cuda(), 0.9)
            loss_offset_52, loss_cls_52, loss_iou_52 = loss_fn(output_52, target_52.cuda(), 0.9)

            loss_offset = loss_offset_13 + loss_offset_26 + loss_offset_52
            loss_cls = loss_cls_13 + loss_cls_26 + loss_cls_52
            loss_iou = loss_iou_13 + loss_iou_26 + loss_iou_52
            loss = 0.5*loss_offset + 0.2*loss_cls + 0.3*loss_iou # 总损失
            # loss = loss_offset + loss_cls + loss_iou

            opt.zero_grad()
            loss.backward()
            opt.step()

            print("epochs:", epochs, " batches:", i)
            print("loss:", loss.item(), "| offset_loss:", loss_offset.item(), "| cls_loss:", loss_cls.item(), "| iou_loss:", loss_iou.item())
            del output_13, output_26, output_52, loss_offset_26, loss_cls_26, loss_offset_52, loss_cls_52, loss_offset, loss_cls, loss, i
            del loss_iou_13, loss_iou_26, loss_iou_52
        epochs += 1
        torch.save(net.state_dict(), "model/darknet.pth")
