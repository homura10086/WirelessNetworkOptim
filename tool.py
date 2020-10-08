
import torch
import numpy as np
import time
import matplotlib.pyplot as plt


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def evaluate_accuracy_2(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.float().to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    train_acc = []
    test_acc = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.float().to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc.append(evaluate_accuracy_2(test_iter, net))
        train_acc.append(train_acc_sum/n)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc[epoch], test_acc[epoch], time.time() - start))

    # 绘图
    plt.plot(range(num_epochs)[:24], test_acc[:24], linewidth=2, color='olivedrab', label='test data')
    plt.plot(range(num_epochs)[:24], train_acc[:24], linewidth=2, color='chocolate', linestyle='--', label='train data')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    test_acc_below = test_acc[:24]
    test_max_below = np.argmax(test_acc_below).item()
    show_max = '[' + str(test_max_below) + ', ' + str(round(test_acc_below[test_max_below], 2)) + ']'
    # 以●绘制最大值点和最小值点的位置
    plt.plot(test_max_below, test_acc_below[test_max_below], 'ko')
    plt.annotate(show_max, xy=(test_max_below, test_acc_below[test_max_below]), xytext=(test_max_below, test_acc_below[test_max_below]))

    plt.grid()
    plt.show()
    plt.plot(range(num_epochs)[25:], test_acc[25:], linewidth=2, color='olivedrab', label='test data')
    plt.plot(range(num_epochs)[25:], train_acc[25:], linewidth=2, color='chocolate', linestyle='--', label='train data')
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    test_acc_above = test_acc[25:]
    test_max_above = np.argmax(test_acc_above).item()
    show_max = '[' + str(test_max_above + 25) + ', ' + str(round(test_acc_above[test_max_above], 2)) + ']'
    # 以●绘制最大值点和最小值点的位置
    plt.plot(test_max_above + 25, test_acc_above[test_max_above], 'ko')
    plt.annotate(show_max, xy=(test_max_above + 25, test_acc_above[test_max_above]), xytext=(test_max_above + 25, test_acc_above[test_max_above]))

    plt.grid()
    plt.show()

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max - min)
