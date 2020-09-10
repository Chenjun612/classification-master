# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):    # 重载索引,对于实例的索引运算，会自动调用__getitem__
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        # print(np.array(image))
        label = self.labels[index]     # 存储标签
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)


class CovidDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, fold, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            fold: five fold index list of dataset
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        # augment = ['00000000', '00100000', '01100000', '10000000', '10100000', '11100000']
        augment = ['00000000']
        fileline = open(image_list_file, "r").readlines()
        for index_num in fold:
            line = fileline[index_num]
            items = line.split()
            image = items[0]
            label = items[1:]
            label = [int(i) for i in label]
            if label[0] == 1:
                for i in augment:
                    image_name = str(int(i) + int(image[:-4])) + '.png'
                    image_name = os.path.join(data_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)
            else:  # label==0
                image_name = os.path.join(data_dir, image)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):    # 重载索引,对于实例的索引运算，会自动调用__getitem__
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        # print(np.array(image))
        label = self.labels[index][0]     # 存储标签
        if self.transform is not None:
            image = self.transform(image)
            # image = (image - image.min()) / (image.max() - image.min())
        return image, label, image_name[-13:-4]

    def __len__(self):
        return len(self.image_names)
