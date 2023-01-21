import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image, ImageDraw

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


def bbox_iou(pred_box, gt_box, xyxy = False, eps = 1e-9):
    box = gt_box.T

    if xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_box[0], pred_box[1], pred_box[2], pred_box[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = gt_box[0], gt_box[1], gt_box[2], gt_box[3]
    else:
        b1_x1, b1_y1 = pred_box[0] - pred_box[2] / 2, pred_box[1] - pred_box[3] / 2
        b1_x2, b1_y2 = pred_box[0] + pred_box[2] / 2, pred_box[1] + pred_box[3] / 2
        b2_x1, b2_y1 = gt_box[0] - gt_box[2] / 2, gt_box[1] - gt_box[3] / 2
        b2_x2, b2_y2 = gt_box[0] + gt_box[2] / 2, gt_box[1] + gt_box[3] / 2

    # Calc intersaction
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) \
            * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Calc Union
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = b1_w * b1_h - b2_w * b2_h - inter + eps

    return inter / union
