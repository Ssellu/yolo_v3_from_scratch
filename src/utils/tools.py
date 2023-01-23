from PIL import Image, ImageDraw

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim


class YOLOV3Props:
    def __init__(self, path) -> None:
        self.cfg_path = path
        self.module_defs = []
        self.set_props()
        self.hyperparameters: dict = self.convert_type(
            type='net', original_dict=self.module_defs[0])

    def convert_type(self, type, original_dict: dict):
        if type == 'net':
            return {
                'batch': int(original_dict['batch']),
                'momentum': float(original_dict['momentum']),
                'decay': float(original_dict['decay']),
                'saturation': float(original_dict['saturation']),
                'learning_rate': float(original_dict['learning_rate']),
                'burn_in': int(original_dict['burn_in']),
                'max_batches': int(original_dict['max_batches']),
                'policy': original_dict['policy'],
                'subdivisions': int(original_dict['subdivisions']),
                'width': int(original_dict['width']),
                'height': int(original_dict['height']),
                'class': int(original_dict['class']),
                'channels': int(original_dict['channels']),
                'ignore_cls': int(original_dict['ignore_cls']),
            }

    def set_props(self):
        raw_str_list = self.parse_hyperparameter_config()
        self.module_defs = []
        for ln in raw_str_list:
            if ln.startswith('['):
                type_name = ln[1:-1]
                di = {'type': type_name}
                if type_name == 'convolutional':
                    di['batch_normalize'] = 0
                self.module_defs.append(di)
            else:
                k, v = ln.split('=')
                k, v = k.strip(), v.strip()
                self.module_defs[-1][k] = v

    def parse_hyperparameter_config(self):
        with open(self.cfg_path, 'r') as file:
            lines = [ln.strip() for ln in file.readlines()
                     if ln.strip() and not ln.startswith('#')]
        return lines

    # parse model layer configuration
    def parse_model_config(self):
        return self.module_defs

def xywh2xyxy_np(x : np.array):
    y = np.zeros_like(x)
    y[...,0] = (x[...,0] - x[...,2]) / 2  # min_x
    y[...,1] = (x[...,1] - x[...,3]) / 2  # min_y
    y[...,2] = (x[...,0] + x[...,2]) / 2  # max_y
    y[...,3] = (x[...,1] + x[...,3]) / 2  # max_y
    return y

def drawBox(img):
    img *= 255
    if img.shape[0]  == 3:
        img_data = np.array(np.transpose(img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(img_data)

    draw = ImageDraw.Draw(img_data)

    plt.imshow(img_data)
    plt.show()


def box_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / (h2+eps)) - torch.atan(w1 / (h1+eps)), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v + eps)
                ciou = iou - (rho2 / c2 + v * alpha)
                return ciou  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def iou(a, b, mode = 0, device = None, eps=1e-9):
    #mode 0 : cxcywh. mode 1 : minmax
    if mode == 0:
        a_x1, a_y1 = a[:,0]-a[:,2]/2, a[:,1]-a[:,3]/2
        a_x2, a_y2 = a[:,0]+a[:,2]/2, a[:,1]+a[:,3]/2
        b_x1, b_y1 = b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2
        b_x2, b_y2 = b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2
    else:
        a_x1, a_y1, a_x2, a_y2 = a[:,0], a[:,1], a[:,2], a[:,3]
        b_x1, b_y1, b_x2, b_y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    xmin = torch.max(a_x1, b_x1)
    xmax = torch.min(a_x2, b_x2)
    ymin = torch.max(a_y1, b_y1)
    ymax = torch.min(a_y2, b_y2)
    #get intersection area
    inter = (xmax - xmin).clamp(0) * (ymax - ymin).clamp(0)
    #get each box area
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1 + eps)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1 + eps)
    union = a_area + b_area - inter + eps

    if device is not None:
        iou = torch.zeros(b.shape[0], device=device)
    else:
        iou = torch.zeros(b.shape[0])
    iou = inter / union

    return iou

def get_lr(optimizer:optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']