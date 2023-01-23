import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from utils.tools import *
from train.loss import *


class Trainer:
    def __init__(self, model: nn.Module, train_loader, eval_loader, hparam, device, torch_writer) -> None:
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

        self.torch_writer = torch_writer

    # Training entry function
    def run(self):
        # for-loop in epoch
        while True:

            if self.max_batch <= self.iter:
                break

            # 1. Train
            self.model.train()

            # 2. Loss calculation
            loss = self.run_iter()

            # 3. Increase index of Epoch
            self.epoch += 1

            # Save model (chekckpoint)
            checkpoint_path = os.path.join('./output', 'model_epoch{}.pth'.format(self.epoch))
            torch.save({'epoch': self.epoch,
                        'iteration':self.iter,
                        'model_state_dict':self.model.state_dict(),
                        'optimizer_state_dict':self.optimizer.state_dict(),
                        'loss':loss}
                        , checkpoint_path=checkpoint_path)



    def run_iter(self):
        for i, batch in enumerate(self.train_loader):

            # Drop the batch when it has invalid values
            if batch is None:
                continue

            input_img, targets, anno_path = batch
            input_img = input_img.to(device=self.device, non_blocking=True)
            # input_img.shape : torch.Size([1, 3, 608, 608]), ==> [batch, channel, image_width, image_height]
            # targets.shape : torch.Size([1, 1, 6]), ==> [1, number of objects, object_info]

            output = self.model(input_img)

            # Get loss between the output and the target
            loss, loss_list = self.yololoss.compute_loss(
                output, targets, self.model.yolo_layers)

            # Get Gradients
            loss.backward()
            self.optimizer.step()                       # Update weights
            self.optimizer.zero_grad()                  # Replace gradients into 0
            # Reset optimizer parameters
            self.scheduler_multistep.step(self.iter)
            self.iter += 1

            # [loss.item(), lobj.item(), lcls.item(), lbox.item()]
            loss_name = ['total_loss', 'obj_loss', 'class_loss', 'box_loss']

            if i % 10 == 0:
                print('epoch : {} / iter : {} / lr : {} / loss : {}'.format(self.epoch,
                      self.iter, get_lr(self.optimizer), loss.item()))
                self.torch_writer.add_scalar(
                    'lr', get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar('total_loss', loss, self.iter)
                for ln, lv in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, lv, self.iter)

        return loss
