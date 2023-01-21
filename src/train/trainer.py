import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from utils.tools import *
from train.loss import *

class Trainer:
    def __init__(self, model: nn.Module, train_loader, eval_loader, hparam, device) -> None:
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam['max_batches']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.yololoss = YoloLoss(self.device, self.model.n_classes)

        self.optimizer = optim.SGD(
            model.parameters(), lr=hparam['learning_rate'], momentum=hparam['momentum'])

        # Learning rate scheduler
        self.scheduler_multistep = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[20, 40, 60], gamma=0.5)

    # Training entry function
    def run(self):
        # for-loop in epoch
        while True:

            # 1. Train
            self.model.train()

            # 2. Loss calculation
            self.run_iter()

            # 3. Increase index of Epoch
            self.epoch += 1

    def run_iter(self):
        for i, batch in enumerate(self.train_loader):

            # Drop the batch when invalid values
            if batch is None:
                continue

            input_img, targets, anno_path = batch
            input_img = input_img.to(device=self.device, non_blocking=True)
            # input_img.shape : torch.Size([1, 3, 608, 608]), ==> [batch, channel, image_width, image_height]
            # targets.shape : torch.Size([1, 1, 6]), ==> [1, number of objects, object_info]


            output = self.model(input_img)

            # Get loss between the output and the target
            self.yololoss.compute_loss(output, targets, self.model.yolo_layers)

            print('!!! output.len : {}, output[0].shape : {}'.format(len(output), output[0].shape))

