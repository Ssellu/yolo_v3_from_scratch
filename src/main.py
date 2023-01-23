import torch
import argparse
import sys
import cv2
from torch.utils.data.dataloader import DataLoader

from utils.tools import *
from dataloader.yolodata import *
from dataloader.data_transforms import *
from model.yolov3 import *
from train.trainer import *

from tensorboardX import SummaryWriter

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

    def collate_fn(batch):
        batch = [data for data in batch if data is not None]

        # Skip in valid data
        if len(batch) == 0:
            return

        images, targets, anno_path = list(zip(*batch))

        images = torch.stack([img for img in images])

        # print('!!! images[0].shape : {}, images.shape : {}'.format(images[0].shape, images.shape))

        for i, boxes in enumerate(targets):
            # set the index of batch
            boxes[:, 0] = i
            print(boxes)
        targets = torch.cat(targets, 0)
        return images, targets, anno_path


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
                                  collate_fn=Main.collate_fn
                                  )


        model = Darknet53(cfg_path=self.args.cfg,
                          param=self.cfg_param, is_train=True)
        model.train()
        model.initialize_weights()

        # Set device
        if torch.cuda.is_available():
            device = torch.device('cuda:0')

        else:
            device = torch.device('cpu')

        model = model.to(device)

        # Load checkpoint
        # If checkpoint exists, load the previous checkpoint.
        if self.args.checkpoint is not None:
            print("Load pretrained model. {}".format(self.args.checkpoint))
            checkpoint = torch.load(self.args.checkpoint, map_location= device)
            # for k, v in checkpoint['model_state_dict'].items():
            #     print(k, v)
            model.load_state_dict(checkpoint['model_state_dict'])

        torch_writer = SummaryWriter("./output")

        train = Trainer(model=model, train_loader=train_loader, eval_loader=None, hparam=self.cfg_param, device=device, torch_writer=torch_writer)
        train.run()


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
