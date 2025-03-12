#score 0:<40,1:40-49,2:50-59,3:60-69,4:70-79,5:80-89,6:90-99
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, images, labels_class, labels_score, transform=None, augment=False):
        self.images = images
        self.labels_class = labels_class
        self.labels_score = labels_score
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label_class = torch.tensor(self.labels_class[idx]).type(torch.long)
        label_score = torch.tensor(self.labels_score[idx]).type(torch.float)

        sample = (img, label_class, label_score)
        return sample

def load_data(path='datasets/diving/diving.csv'):
    diving = pd.read_csv(path)
    diving_mapping = {0: '2.8.305C', 1: '2.8.405B', 2: '2.9.205B', 3: '2.9.5152B', 4: '3.0.107B', 5: '3.0.305B', 6: '3.2.407C',7: '3.2.5253B', 8: '3.2.6243D', 9: '3.3.207C', 10: '3.3.626C'}

    return diving, diving_mapping

def prepare_data(data):
    """ Prepare data for multi-task learning:
        input: data frame with labels and pixel data
        output: image and two different label arrays (classification and regression) """
    

    image_array = np.zeros(shape=(len(data), 48, 48))


    image_label_class = np.array(list(map(int, data['emotion'])))
    

    image_label_score = np.array(list(map(float, data['score'])))  
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label_class, image_label_score

def get_dataloaders(path='datasets/diving/diving.csv', bs=64, augment=True):
    """ Prepare train, val, & test dataloaders for multi-task learning """
    diving, diving_mapping = load_data(path)

    # 准备数据
    xtrain, ytrain_class, ytrain_score = prepare_data(diving[diving['Usage'] == 'Training'])
    xval, yval_class, yval_score = prepare_data(diving[diving['Usage'] == 'PrivateTest'])
    xtest, ytest_class, ytest_score = prepare_data(diving[diving['Usage'] == 'PublicTest'])

    mu, st = 0, 255

    test_transform = transforms.Compose([ 
        transforms.Grayscale(),
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]))])

    if augment:
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.FiveCrop(40),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.RandomErasing()(t) for t in tensors]))])
    else:
        train_transform = test_transform

    # 创建数据集
    train = CustomDataset(xtrain, ytrain_class, ytrain_score, train_transform)
    val = CustomDataset(xval, yval_class, yval_score, test_transform)
    test = CustomDataset(xtest, ytest_class, ytest_score, test_transform)

    # 创建数据加载器
    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=4)
    valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=4)
    testloader = DataLoader(test, batch_size=64, shuffle=True, num_workers=4)

    return trainloader, valloader, testloader

# class CustomDataset(Dataset):
#     def __init__(self, images, labels, transform=None, augment=False):
#         self.images = images
#         self.labels = labels
#         self.transform = transform

#         self.augment = augment

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img = np.array(self.images[idx])

#         img = Image.fromarray(img)

#         if self.transform:
#             img = self.transform(img)

#         label = torch.tensor(self.labels[idx]).type(torch.long)
#         sample = (img, label)

#         return sample


# def load_data(path='datasets/diving/diving.csv'):
#     diving = pd.read_csv(path)
#     diving_mapping = {0: '2.8.305C', 1: '2.8.405B', 2: '2.9.205B', 3: '2.9.5152B', 4: '3.0.107B', 5: '3.0.305B', 6: '3.2.407C',7: '3.2.5253B', 8: '3.2.6243D', 9: '3.3.207C', 10: '3.3.626C'}

#     return diving, diving_mapping


# def prepare_data(data):
#     """ Prepare data for modeling
#         input: data frame with labels und pixel data
#         output: image and label array """

#     image_array = np.zeros(shape=(len(data), 48, 48))
#     image_label = np.array(list(map(int, data['emotion'])))

#     for i, row in enumerate(data.index):
#         image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
#         image = np.reshape(image, (48, 48))
#         image_array[i] = image

#     return image_array, image_label


# def get_dataloaders(path='datasets/diving/diving.csv', bs=64, augment=True):
#     """ Prepare train, val, & test dataloaders
#         Augment training data using:
#             - cropping
#             - shifting (vertical/horizental)
#             - horizental flipping
#             - rotation
#         input: path to diving csv file
#         output: (Dataloader, Dataloader, Dataloader) """

#     diving, diving_mapping = load_data(path)

#     xtrain, ytrain = prepare_data(diving[diving['Usage'] == 'Training'])
#     xval, yval = prepare_data(diving[diving['Usage'] == 'PrivateTest'])
#     xtest, ytest = prepare_data(diving[diving['Usage'] == 'PublicTest'])

#     mu, st = 0, 255

#     test_transform = transforms.Compose([
#         transforms.Grayscale(),
#         transforms.TenCrop(40),
#         transforms.Lambda(lambda crops: torch.stack(
#             [transforms.ToTensor()(crop) for crop in crops])),
#         transforms.Lambda(lambda tensors: torch.stack(
#             [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
#     ])
#     if augment:
#         train_transform = transforms.Compose([
#             transforms.Grayscale(),
#             transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
#             transforms.RandomApply([transforms.ColorJitter(
#                 brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
#             transforms.RandomApply(
#                 [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
#             transforms.FiveCrop(40),
#             transforms.Lambda(lambda crops: torch.stack(
#                 [transforms.ToTensor()(crop) for crop in crops])),
#             transforms.Lambda(lambda tensors: torch.stack(
#                 [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
#             transforms.Lambda(lambda tensors: torch.stack(
#                 [transforms.RandomErasing()(t) for t in tensors])),
#         ])
#     else:
#         train_transform = test_transform

#     # X = np.vstack((xtrain, xval))
#     # Y = np.hstack((ytrain, yval))

#     train = CustomDataset(xtrain, ytrain, train_transform)
#     val = CustomDataset(xval, yval, test_transform)
#     test = CustomDataset(xtest, ytest, test_transform)

#     trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=4)
#     valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=4)
#     testloader = DataLoader(test, batch_size=64, shuffle=True, num_workers=4)

#     return trainloader, valloader, testloader
