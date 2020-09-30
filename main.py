

import pandas as pd
import torch
from torch import nn
import torch.utils.data as Data
from torch.nn import init
import tool

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.ReLU = nn.ReLU()
    def forward(self, x):
        y1 = self.linear(x)
        y2 = self.ReLU(y1)
        y = self.linear2(y2)
        return y

seed = 0
torch.manual_seed(seed)

# num_feature = 3
# num_label = 6
# batch_size = 10
# lr = 0.008
# num_epochs = 30
# num_hidden = 6
# feature = pd.read_csv('data.csv', header=0, usecols=range(num_feature))
# label = pd.read_csv('data.csv', header=0, usecols=[num_feature])

num_feature = 10
num_label = 5
batch_size = 100
lr = 0.05
num_epochs = 10
num_hidden = 10
feature = pd.read_csv('NewData3.csv', header=0, usecols=range(num_feature))
label = pd.read_csv('NewData3.csv', header=0, usecols=[num_feature])

features = torch.Tensor(feature.values)
labels = torch.LongTensor(label.values).squeeze()

# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
num_example = len(dataset)
num_train = int(0.8 * num_example)
num_test = num_example - num_train
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])

# 把 dataset 放入 DataLoader
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

net = LinearNet(num_feature, num_label, num_hidden)

# 神经网络参数初始化
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]

    test_acc = tool.evaluate_accuracy(test_iter, net)

    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
          % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
