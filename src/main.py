import torch
import argparse, sys

from utils.tools import *

class Main:
    def __init__(self) -> None:
        args = Main.parse_args()
        props = YOLOV3Props(self.args.cfg)

        self.cfg_param = props.hyperparameters
        self.using_gpus = [int(g) for g in args.gpus]

    def parse_args():
        parser = argparse.ArgumentParser(description="YOLOV3_PYTORCH Arguements")
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

    def train(self):pass
    def eval(self):pass
    def demo(self):pass

    def main(self):
        if self.args.mode == 'train':
            self.train()
        elif self.args.mode == 'eval':
            self.eval()
        elif self.args.mode == 'demo':
            self.demo()


if __name__ == '__main__':
    Main().main()
