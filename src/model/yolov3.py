import os
import sys
import numpy as np
import torch
import torch.nn as nn

from utils.tools import *


class Darknet53(nn.Module):
    def __init__(self, cfg_path, param, is_train):
        super().__init__()
        self.is_train = is_train
        self.batch = int(param['batch'])
        self.in_channel = int(param['channels'])
        self.in_width = int(param['width'])
        self.in_height = int(param['height'])
        self.n_classes = int(param['class'])
        self.module_cfg = YOLOV3Props(cfg_path).parse_model_config()[1:]
        self.module_list = self.get_layers()

    def get_layers(self):
        module_list = nn.ModuleList()
        layer_info = self.module_cfg
        in_channels = [self.in_channel]  # First channels of input
        for layer_idx, info in enumerate(layer_info):
            modules = nn.Sequential()
            if info['type'] == 'convolutional':
                Darknet53.make_conv_layer(
                    layer_idx, modules, info, in_channel=in_channels[-1])
                in_channels.append(int(info['filters']))
            elif info['type'] == 'shortcut':
                Darknet53.make_shortcut_layer(layer_idx, modules)
                in_channels.append(in_channels[-1])
            elif info['type'] == 'route':  # Concatenation
                Darknet53.make_route_layer(layer_idx, modules)
                layers = [int(y) for y in info['layers'].split(', ')]
                ln = len(layers)
                if ln == 1:
                    in_channels.append(in_channels[layers[0]])
                elif ln == 2:
                    in_channels.append(
                        in_channels[layers[0]] + in_channels[layers[1]])
            elif info['type'] == 'upsample':  #
                Darknet53.make_upsample_layer(layer_idx, modules, info)
                in_channels.append(in_channels[-1])
            elif info['type'] == 'yolo':
                yolo_layer = YoloLayer(
                    info, self.in_width, self.in_height, self.is_train)
                modules.add_module(
                    'layer_{}_yolo'.format(layer_idx), yolo_layer)
                in_channels.append(in_channels[-1])
            module_list.append(modules)
        return module_list

    def initialize_weights(self):
        # Track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # Scale
                nn.init.constant_(m.bias, 0)    # Shift

            elif isinstance(m, nn.Linear):  # FC
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        yolo_result = []
        layer_result = []

        # Important!

        for idx, (name, layer) in enumerate(zip(self.module_cfg, self.module_list)):
            if name['type'] == 'convolutional':
                x = layer(x)
                layer_result.append(x)

            elif name['type'] == 'shortcut':
                x += layer_result[int(name['from'])]
                layer_result.append(x)
            elif name['type'] == 'yolo':
                yolo_x = layer(x)
                layer_result.append(yolo_x)
                yolo_result.append(yolo_x)
            elif name['type'] == 'upsample':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'route':
                layers = [int(y) for y in name['layers'].split(', ')]
                x = torch.cat([layer_result[layer] for layer in layers], dim=1)
                layer_result.append(x)
        return yolo_result

    def make_conv_layer(layer_idx: int, modules: nn.Module, layer_info: dict, in_channel: int):
        filters = int(layer_info['filters'])  # Size of the output channel
        size = int(layer_info['size'])  # Size of the kernel
        stride = int(layer_info['stride'])
        pad = size // 2

        if layer_info['batch_normalize'] == '1':
            modules.add_module(name='layer_{}_conv'.format(layer_idx),
                               module=nn.Conv2d(in_channels=in_channel,
                                                out_channels=filters,
                                                kernel_size=size,
                                                stride=stride,
                                                padding=pad))

            modules.add_module(
                name='layer_{}_bn'.format(layer_idx),
                module=nn.BatchNorm2d(num_features=filters))
        else:
            modules.add_module(name='layer_{}_conv'.format(layer_idx),
                               module=nn.Conv2d(in_channels=in_channel,
                                                out_channels=filters,
                                                kernel_size=size,
                                                stride=stride,
                                                padding=pad))
        if layer_info['activation'] == 'leaky':
            modules.add_module(
                name='layer_{}_act'.format(layer_idx),
                module=nn.LeakyReLU())
        elif layer_info['activation'] == 'relu':
            modules.add_module(
                name='layer_{}_act'.format(layer_idx),
                module=nn.ReLU())

    def make_shortcut_layer(layer_idx: int, modules: nn.Module):
        modules.add_module(
            'layer_{}_shortcut'.format(layer_idx),
            nn.Identity())

    def make_route_layer(layer_idx: int, modules: nn.Module):
        modules.add_module(
            'layer_{}_route'.format(layer_idx),
            nn.Identity())

    def make_upsample_layer(layer_idx: int, modules: nn.Module, layer_info: dict):
        modules.add_module(
            name='layer_{}_upsample'.format(layer_idx),
            module=nn.Upsample(scale_factor=int(layer_info['stride']), mode='nearest'))


class YoloLayer(nn.Module):
    def __init__(self, layer_info: dict, in_width: int, in_height: int, is_train: bool):
        super(YoloLayer, self).__init__()
        self.n_classes = int(layer_info['classes'])
        self.ignore_thresh = float(layer_info['ignore_thresh'])
        # bounding_box[4] + objectness[1] + class_probability[n_classes]
        self.box_attr_size = self.n_classes + 5

        mask_indices = [int(x) for x in layer_info['mask'].split(',')]  # 1 3 5
        anchor_all = [[int(y) for y in x.split(',')]
                      for x in layer_info['anchors'].split(', ')]
        self.anchor = torch.tensor([anchor_all[x] for x in mask_indices])
        self.in_width = in_width
        self.in_height = in_height
        self.stride = None
        self.lw = None
        self.lh = None
        self.is_train = is_train

    def forward(self, x):
        # x : input array [N, C, H, W]
        self.lw, self.lh = x.shape[3], x.shape[2]
        self.anchor = self.anchor.to(x.device)
        self.stride = torch.tensor([torch.div(self.in_width, self.lw, rounding_mode='floor'),
                                    torch.div(self.in_height, self.lh, rounding_mode='floor')]).to(x.device)

        # For KITTIE data set, n_classes is 8
        # Therefore, C = (8 + 5) * 3 = 39
        #                 8 is the mean value of the threshhold
        #                     5 is value of objectness
        #                          3 is the number of the anchors
        # Input shape of image x is [batch, box_attr * anchor, lw, lh]
        #  e.g. [1, 39, 19, 19]

        # 4-dim [batch, box_attr * anchor, lh, lw]
        #   -> 5-dim [batch, ahchor, box_attr, lh, lw]
        #       -> [batch, anchor, lh, lw, box_attr]
        print(x.shape)
        x = x.view(-1, self.anchor.shape[0], self.box_attr_size,
                   self.lh, self.lw).permute(0, 1, 3, 4, 2).contiguous()
        return x
