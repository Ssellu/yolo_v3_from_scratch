import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

class YOLOV3Props:
    def __init__(self, path) -> None:
        self.module_defs = []
        self.set_props(path)

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

    def set_props(self, path):
        raw_str_list = self.parse_hyperparameter_config_from(path)
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

    def parse_hyperparameter_config_from(self, path):
        with open(path, 'r') as file:
            lines = [ln.strip() for ln in file.readlines()
                     if ln.strip() and not ln.startswith('#')]
        return lines

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
