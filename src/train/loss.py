import os, sys

import torch
import torch.nn as nn
from utils.tools import *

class YoloLoss(nn.Module):
    def __init__(self, device, num_class) -> None:
        super(YoloLoss, self).__init__()
        self.device = device
        self.num_class = num_class
        self.mseloss = nn.MSELoss().to(self.device)
        self.bceloss = nn.BCELoss().to(self.device)  # BinaryCrossEntropy
        self.bcelogloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device)).to(device)  # Logit BinaryCrossEntropy

    def compute_loss(self, pred, targets, yololayer):
        # Bounding box Loss
        lbox = torch.zeros(1, device=self.device)
        # Class Loss
        lcls = torch.zeros(1, device=self.device)
        # Objectness Loss
        lobj = torch.zeros(1, device=self.device)

        # Get positive targets
        t_cls, t_boxes, t_indices, t_anchors = self.get_targets(pred, targets, yololayer)

        # loop for predicted 3 yolo layers
        for p_idx, p_out in enumerate(pred):
            # print('!!! Yolo : {}, shape : {}'.format(p_idx, p_out.shape))
            # !!! Yolo : 0, shape : torch.Size([1, 3, 19, 19, 13])
            # !!! Yolo : 1, shape : torch.Size([1, 3, 38, 38, 13])
            # !!! Yolo : 2, shape : torch.Size([1, 3, 76, 76, 13])
            # --> p_out.shape : [batch, anchors, grid_y, grid_x, box_attributes]
            # --> The number of boxes in each yolo layer : anchors * grid_height * grid_width
            #           Yolo0 -> 3 * 19 * 19
            #           Yolo1 -> 3 * 38 * 38
            #           Yolo2 -> 3 * 76 * 76

            # Positive prediction vs Negative prediction
            # pos : nag = 0.01 : 0.99

            # We can get box_loss and class_loss in positive prediction,
            # and obj_loss in negative prediction

            batch_id, anchor_id, gy, gx = t_indices[p_idx]

            t_obj = torch.zeros_like(p_out[..., 0], device=self.device)
            num_targets = batch_id.shape[0]

            if num_targets:
                ps = p_out[batch_id, anchor_id, gy, gx]  # [batch, anchor, grid_h, grid_w, box_attr]
                pxy = torch.sigmoid(ps[..., 0:2])
                pwh = torch.exp(ps[...,2:4]) * t_anchors[p_idx]
                pbox = torch.cat((pxy, pwh), dim=1)


                print(f'pbox.shape : {pbox.shape} / t_boxes[p_idx].shape : {t_boxes[p_idx].shape}')
                iou = bbox_iou(pbox.T, t_boxes[p_idx])


                # Box Loss
                # MSE
                # loss_xy = self.mseloss(pbox[..., 0:2], t_boxes[p_idx][..., 0:2])
                # loss_wh = self.mseloss(pbox[..., 2:4], t_boxes[p_idx][..., 2:4])
                lbox += (1 - iou).mean()

                # Objectness Loss
                # GT box and predicted box -> positive : 1 / negative -> 0 (Using IOU)
                t_obj[batch_id, anchor_id, gy, gx] = iou.detach().clamp(0).type(t_obj.dtype)


                # Class Loss
                if ps.size(1) - 5 > 1:
                    t = torch.zeros_like(ps[..., 5:], device=self.device)
                    t[range(num_targets), t_cls[p_idx]] = 1
                    # print("cls")
                    # print(t)

                    lcls += self.bcelogloss(ps[:, 5:], t)
                    # print("ps")
                    # print(ps[:, 5:])

            lobj += self.bcelogloss(p_out[..., 4], t_obj)

        # Loss Weight
        lcls *= 0.05
        lobj *= 1.0
        lbox *= 0.5

        # Total Loss
        loss = lcls + lbox + lobj
        loss_list = [loss.item(), lobj.item(), lcls.item(), lbox.item()]

        return loss, loss_list


    def get_targets(self, preds, targets:np.ndarray, yololayer):
        num_anchors = 3
        num_targets = targets.shape[0]
        tcls, tboxes, indices, anch = [], [], [], []

        gain = torch.ones(7, device=self.device)
        ai = torch.arange(num_anchors, device=self.device).float().view(num_anchors, 1).repeat(1, num_targets)
        # torch.arange(num_anchors, device=self.device).float() --> [0., 1., 2.]
        # .view() --> [0.
        #              1.
        #              2.]
        # .repeat(1, 3) --> [[0. 0. 0.]
        #                    [1. 1. 1.]
        #                    [2. 2. 2.]]

        # [batch_id, class_id, box_cx, box_cy, box_w, box_h, anchor_id]
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)

        for yi, yl in enumerate(yololayer):
            anchors = yl.anchor / yl.stride
            gain[2:6] = torch.tensor(preds[yi].shape)[[3,2,3,2]]  # grid_w, grid_h

            t = targets * gain

            # Pick the best anchor
            if num_targets:
                r = t[:, :, 4:6] / anchors[:, None]

                # Select the ratios less than 4
                j = torch.max(r, 1. / r).max(dim=2)[0] < 4

                t = t[j]

            else:
                t = targets[0]

            # b : batch id
            # c : class id
            b, c = t[:, :2].long().T

            # target's x, y
            txy = t[:, 2:4]
            twh = t[:, 4:6]

            # Cx Cy
            tij = txy.long()
            ti, tj = tij.T

            # Anchor Index
            a = t[:, 6].long()

            # Add indices
            indices.append([b, a, tj.clamp_(0, gain[3]-1), ti.clamp_(0, gain[2]-1)])


            # Add target box
            tboxes.append(torch.cat((txy - tij, twh), dim=1))

            # Add anchor
            anch.append(anchors[a])

            # Add class
            tcls.append(c)

        return tcls, tboxes, indices, anch

