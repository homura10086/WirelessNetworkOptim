
import pandas as pd
import torch
import torch.utils.data as Data
from torch import nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import tool


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 3),  # in_channels, out_channels, kernel_size 输入28*28 输出24*24 new输入18*10
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride 输入24*24 输出12*12
            nn.Conv2d(6, 12, 3),  # 输入12*12 输出8*8
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)  # 输入8*8 输出4*4 通道维16
        )
        self.fc = nn.Sequential(
            nn.Linear(36, 12),  # 全连接层从16*4*4降维到10
            nn.Sigmoid(),
            nn.Linear(12, 5)
        )

    def forward(self, x):
        output1 = self.conv(x)
        output = self.fc(output1.view(x.shape[0], -1))
        return output

num_RAN = 10000
num_cell = 18
num_feature = 10
rate_test = 0.2
batch_size = 256
lr = 0.001
num_epochs = 50
device = 'cuda'

# 数据处理
feature = pd.read_csv('NewData5.csv', header=0, usecols=range(num_feature))
label = pd.read_csv('NewData5.csv', header=0, usecols=[num_feature])
feature_normalize = np.zeros((num_RAN * num_cell, num_feature))
for i in range(num_feature):
    operation_feature = np.array(feature[feature.columns[i]])
    feature_normalize[:, i] = tool.minmaxscaler(operation_feature)
features = torch.from_numpy(feature_normalize).reshape(num_RAN, 1, num_cell, num_feature)
label = torch.LongTensor(label.values).squeeze()
labels = torch.zeros(10000, dtype=int)
for i in range(10000):
    labels[i] = label[i * 18]

# 数据集处理
dataset = Data.TensorDataset(features, labels)
num_test = int(rate_test * num_RAN)
num_train = num_RAN - num_test
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
train_iter = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)
test_iter = Data.DataLoader(
    dataset=test_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

net = LeNet()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
tool.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
