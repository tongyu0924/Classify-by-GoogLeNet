import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
from GoogLeNet import GoogLeNet
import matplotlib.pyplot as plt

def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 224, 224, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (224, 224))
        if label:
            y[i] = int(file.split("_")[0])

    if label:
        return x, y
    else:
        return x



path = 'food-11/food-11'
train_x, train_y = readfile(os.path.join(path, "training"), True)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = self.x[idx]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[idx]
            return X, Y
        else:
            return X

batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

model = GoogLeNet(11)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 1

result_acc = []
result_loss = []

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0])
        batch_loss = loss(train_pred, data[1])
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.data.numpy(), axis=1) == data[1].numpy())
        train_loss = batch_loss.item()
        result_acc.append(train_acc / data[0].size(0))
        result_loss.append(train_loss)


plt.figure(figsize=(8, 8))
plt.plot(result_acc)
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(result_loss)
plt.show()

