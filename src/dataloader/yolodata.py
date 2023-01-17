import sys
import os
from PIL import Image

import torch
import numpy as np

from torch.utils.data import Dataset


class Yolodata(Dataset):

    root_dataset_path = 'dataset'
    class_str = ['Car', 'Van', 'Truck', 'Pedestrian',
                 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

    def __init__(self, is_train=True, transform=None, cfg_param=None) -> None:
        super(Yolodata, self).__init__()

        self.load_type = 'train' if is_train else 'eval'
        self.transform = transform

        self.image_dir = '{}/{}/png_images/'.format(
            self.root_dataset_path, self.load_type)
        self.annotation_dir = '{}/{}/annotations/'.format(
            self.root_dataset_path, self.load_type)
        self.file_txt = '{}/{}/image_sets/{}.txt'.format(
            self.root_dataset_path, self.load_type, self.load_type)

        self.num_class = cfg_param['class']
        self.img_data = []
        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as file:
            lines = file.readlines()
            self.img_names = [i.strip() for i in lines]

        for i in self.img_names:
            if os.path.exists('{}{}.jpg'.format(self.image_dir, i)):
                self.img_data.append(i + '.jpg')
            elif os.path.exists('{}{}.jpeg'.format(self.image_dir, i)):
                self.img_data.append(i + '.jpeg')
            elif os.path.exists('{}{}.png'.format(self.image_dir, i)):
                self.img_data.append(i + '.png')

    # Get an item for a iteration in a batch
    def __getitem__(self, index):
        img_path = '{}{}.png'.format(self.image_dir, self.img_names[index])
        if not os.path.exists(img_path):
            raise FileNotFoundError('Invalid Image Index.')
        with open(img_path, 'rb') as file:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            # Image.shape : [H, W, C]
            img_origin_h, img_origin_w = img.shape[:2]


        annotation_path = '{}{}.txt'.format(
            self.annotation_dir, self.img_names[index])
        bounding_box = []

        if not os.path.exists(self.annotation_dir):
            if self.transform is not None:
                img = self.transform((img, np.array((0, 0, 0, 0, 0))))[0]

            return img, None, None

        with open(annotation_path, 'r') as file:
            for line in file.readlines():
                bounding_box = np.array([[float(n)] for n in line.split(' ')]).reshape(1, 5)

            # Skip empty target, 0, 0, 0, 0)]).shape))
            empty_target = False
            if bounding_box.shape[0] == 0:
                empty_target = True
                bounding_box = np.array([(0, 0, 0, 0, 0)])

            # Data augmentation
            if self.transform is not None:
                img, bounding_box = self.transform((img, bounding_box))

            if empty_target:
                return   # TODO raise?

            batch_idx = torch.zeros(bounding_box.shape[0])


            # TODO Switch into this below
            # target_data = torch.cat(
            #     (batch_idx.view(-1, 1), bounding_box), dim=1)

            target_data = torch.cat(
                (batch_idx.view(-1, 1), torch.tensor(bounding_box).clone().detach()), dim=1)

            return img, target_data, annotation_path

    def __len__(self):
        return len(self.img_data)