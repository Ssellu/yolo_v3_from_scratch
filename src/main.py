import torch
import argparse
import sys
import cv2
from torch.utils.data.dataloader import DataLoader

from utils.tools import *
from dataloader.yolodata import *
from dataloader.data_transforms import *
from model.yolov3 import *


class Main:
    def __init__(self) -> None:
        self.args = Main.parse_args()
        self.cfg = YOLOV3Props(self.args.cfg)
        self.cfg_param = self.cfg.hyperparameters
        self.using_gpus = [int(g) for g in self.args.gpus]

    def parse_args():
        parser = argparse.ArgumentParser(
            description="YOLOV3_PYTORCH Arguements")
        parser.add_argument("--gpus", type=int, nargs='+',
                            default=[], help="List of GPU device ID")
        parser.add_argument("--mode", type=str, default=None,
                            help="Mode : train / eval / demo")
        parser.add_argument("--cfg", type=str, default=None,
                            help="Model config file path")
        parser.add_argument("--checkpoint", type=str, default=None,
                            help="Model checkpoint path")

        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(1)

        args = parser.parse_args()
        return args

    def train(self):
        my_transform = get_transformations(
            cfg_param=self.cfg_param, is_train=True)
        train_data = Yolodata(is_train=True,
                              transform=my_transform,
                              cfg_param=self.cfg_param)

        # DataLoader for 6081 images with 4 batches
        train_loader = DataLoader(train_data,
                                  batch_size=self.cfg_param['batch'],
                                  num_workers=0,
                                  pin_memory=True,  # Fix the location of image on memory
                                  drop_last=True,
                                  shuffle=True,
                                  # TODO collate_fn=?
                                  )

        model = Darknet53(cfg_path=self.args.cfg,
                          param=self.cfg_param, is_train=True)

        model.train()
        for i, batch in enumerate(train_loader):
            img, targets, annotation_path = batch
            output = model(img)
            print("shape {} {} {}".format(output[0].shape, output[1].shape, output[2].shape,))


    def eval(self):
        Yolodata(is_train=False, cfg_param=self.cfg_param)

    def demo(self):
        pass

    def main(self):
        if self.args.mode == 'train':
            self.train()
        elif self.args.mode == 'eval':
            self.eval()
        elif self.args.mode == 'demo':
            self.demo()


if __name__ == '__main__':
    Main().main()
