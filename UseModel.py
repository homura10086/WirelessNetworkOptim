
import pandas as pd
import torch
import torch.utils.data as Data
from torch import nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import tool
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 3),  # in_channels, out_channels, kernel_size 输入28*28 输出24*24 new输入18*10
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride 输入24*24 输出12*12
        )
        self.fc = nn.Sequential(
            nn.Linear(48, 24),
            nn.Sigmoid(),
            nn.Linear(24, 12),
            nn.Sigmoid(),
            nn.Linear(12, 5)
        )

    def forward(self, x):
        output1 = self.conv(x)
        # output = self.fc(x.view(x.shape[0], -1))
        output = self.fc(output1.view(x.shape[0], -1))
        return output

def index_generate(num):
    data_acc = int(0)
    index = torch.zeros(batch_size).view(-1, 1).to(device)
    for i in range(batch_size):
        index[i][0] = num
    index = index.long()
    return index

seed = 1
torch.manual_seed(seed)

num_RAN = 30000
num_cell = 6
num_feature = 10
batch_size = 30000
device = 'cuda'

# 模型加载
net = Net()
net.load_state_dict(torch.load('./NetParam.pkl'))

# 数据处理for Data1007
feature = pd.read_csv('Data1007.csv', header=0, usecols=range(num_feature))
label = pd.read_csv('Data1007.csv', header=0, usecols=[num_feature])
feature_normalize = np.zeros((num_RAN * num_cell, num_feature))
for i in range(num_feature):
    operation_feature = np.array(feature[feature.columns[i]])
    feature_normalize[:, i] = tool.minmaxscaler(operation_feature)
features = torch.from_numpy(feature_normalize).reshape(num_RAN, 1, num_cell, num_feature)
label = torch.LongTensor(label.values).squeeze()
labels = torch.zeros(num_RAN, dtype=int)
for i in range(num_RAN):
    for j in range(6):
        if label[i * num_cell + j] != 0:
            labels[i] = label[i * num_cell + j]
            break
        if j == 5:
            labels[i] = 0

# 数据集处理
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

net = net.to(device)
data_acc1, data_acc2, data_acc3, data_acc4, data_acc5, n = 0, 0, 0, 0, 0, 0
data_acc11, data_acc21, data_acc31, data_acc41, data_acc51, n1 = 0, 0, 0, 0, 0, 0
data_acc12, data_acc22, data_acc32, data_acc42, data_acc52, n2 = 0, 0, 0, 0, 0, 0
data_acc13, data_acc23, data_acc33, data_acc43, data_acc53, n3 = 0, 0, 0, 0, 0, 0
data_acc14, data_acc24, data_acc34, data_acc44, data_acc54, n4 = 0, 0, 0, 0, 0, 0

# y标签 y_hat_order预测标签概率从小到大排列
# 求y中各问题个数

for X, y in data_iter:
    X = X.float().to(device)
    y = y.to(device)
    y_hat = net(X)
    y_hat_softmax = F.softmax(y_hat, dim=1)  # 各样本概率分布
    y_hat_order = y_hat_softmax.argsort(dim=1)  # 各样本概率分布从小到大的索引排列
    y_hat_order_max1 = torch.gather(y_hat_order, dim=1, index=index_generate(4)).view(batch_size)
    y_hat_order_max2 = torch.gather(y_hat_order, dim=1, index=index_generate(3)).view(batch_size)
    y_hat_order_max3 = torch.gather(y_hat_order, dim=1, index=index_generate(2)).view(batch_size)
    y_hat_order_max4 = torch.gather(y_hat_order, dim=1, index=index_generate(1)).view(batch_size)
    y_hat_order_max5 = torch.gather(y_hat_order, dim=1, index=index_generate(0)).view(batch_size)
    for i in range(batch_size):
        if y[i] == 0:
            n = n + 1
            if y_hat_order_max1[i] == 0:
                data_acc1 = data_acc1 + 1
            elif y_hat_order_max2[i] == 0:
                data_acc2 = data_acc2 + 1
            elif y_hat_order_max3[i] == 0:
                data_acc3 = data_acc3 + 1
            elif y_hat_order_max4[i] == 0:
                data_acc4 = data_acc4 + 1
            elif y_hat_order_max5[i] == 0:
                data_acc5 = data_acc5 + 1
        elif y[i] == 1:
            n1 = n1 + 1
            if y_hat_order_max1[i] == 1:
                data_acc11 = data_acc11 + 1
            elif y_hat_order_max2[i] == 1:
                data_acc21 = data_acc21 + 1
            elif y_hat_order_max3[i] == 1:
                data_acc31 = data_acc31 + 1
            elif y_hat_order_max4[i] == 1:
                data_acc41 = data_acc41 + 1
            elif y_hat_order_max5[i] == 1:
                data_acc51 = data_acc51 + 1
        elif y[i] == 2:
            n2 = n2 + 1
            if y_hat_order_max1[i] == 2:
                data_acc12 = data_acc12 + 1
            elif y_hat_order_max2[i] == 2:
                data_acc22 = data_acc22 + 1
            elif y_hat_order_max3[i] == 2:
                data_acc32 = data_acc32 + 1
            elif y_hat_order_max4[i] == 2:
                data_acc42 = data_acc42 + 1
            elif y_hat_order_max5[i] == 2:
                data_acc52 = data_acc52 + 1
        elif y[i] == 3:
            n3 = n3 + 1
            if y_hat_order_max1[i] == 3:
                data_acc13 = data_acc13 + 1
            elif y_hat_order_max2[i] == 3:
                data_acc23 = data_acc23 + 1
            elif y_hat_order_max3[i] == 3:
                data_acc33 = data_acc33 + 1
            elif y_hat_order_max4[i] == 3:
                data_acc43 = data_acc43 + 1
            elif y_hat_order_max5[i] == 3:
                data_acc53 = data_acc53 + 1
        else:
            n4 = n4 + 1
            if y_hat_order_max1[i] == 4:
                data_acc14 = data_acc14 + 1
            elif y_hat_order_max2[i] == 4:
                data_acc24 = data_acc24 + 1
            elif y_hat_order_max3[i] == 4:
                data_acc34 = data_acc34 + 1
            elif y_hat_order_max4[i] == 4:
                data_acc44 = data_acc44 + 1
            elif y_hat_order_max5[i] == 4:
                data_acc54 = data_acc54 + 1

print('y = 0')
print(data_acc1 / n)
print(data_acc2 / n)
print(data_acc3 / n)
print(data_acc4 / n)
print(data_acc5 / n)
print('y = 1')
print(data_acc11 / n1)
print(data_acc21 / n1)
print(data_acc31 / n1)
print(data_acc41 / n1)
print(data_acc51 / n1)
print('y = 2')
print(data_acc12 / n2)
print(data_acc22 / n2)
print(data_acc32 / n2)
print(data_acc42 / n2)
print(data_acc52 / n2)
print('y = 3')
print(data_acc13 / n3)
print(data_acc23 / n3)
print(data_acc33 / n3)
print(data_acc43 / n3)
print(data_acc53 / n3)
print('y = 4')
print(data_acc14 / n4)
print(data_acc24 / n4)
print(data_acc34 / n4)
print(data_acc44 / n4)
print(data_acc54 / n4)

