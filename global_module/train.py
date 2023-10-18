import time
import torch
import numpy as np
import sys
sys.path.append('../global_module/')
import global_module.d2lzh_pytorch as d2l
import matplotlib.pyplot as plt
from pylab import *

def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0

    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            X = X.to(device)
            y = y.to(device)
            net.eval() # 评估模式, 这会关闭dropout
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.long()).sum().cpu().item()
            test_l_sum += l
            test_num += 1
            net.train() # 改回训练模式
            n += y.shape[0]
    new_acc = acc_sum / n

    return [new_acc, test_l_sum]

def train(net, train_iter, valida_iter, loss, optimizer, device, epochs=30, early_stopping=True,
          early_num=20):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    best_acc=0
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        #原来耐心值5
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,15, eta_min=0.0, last_epoch=-1)
        #print(optimizer.state_dict()['param_groups'][0]['lr'])
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            # print('y_hat', y_hat)
            # print('y', y)
            l = loss(y_hat, y.long())#损失值计算

            optimizer.zero_grad()#梯度清零
            l.backward()##反向传播
            optimizer.step()##更新梯度
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y.long()).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step(epoch)

        acc_sum, n = 0.0, 0

        with torch.no_grad():
            for X, y in valida_iter:
                test_l_sum, test_num = 0, 0
                X = X.to(device)
                y = y.to(device)
                net.eval()  # 评估模式, 这会关闭dropout
                y_hat = net(X)
                l = loss(y_hat, y.long())
                acc_sum += (y_hat.argmax(dim=1) == y.long()).sum().cpu().item()
                test_l_sum += l
                test_num += 1
                net.train()  # 改回训练模式
                n += y.shape[0]
        new_acc = acc_sum / n

        valida_acc = new_acc
        valida_loss = test_l_sum

      #  valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        loss_list.append(valida_loss)
        if valida_acc > best_acc:
            best_acc = valida_acc

        train_loss_list.append(train_l_sum) # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(best_acc)
        #print('v :',valida_acc_list)
        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, valida_loss, best_acc, time.time() - time_epoch))

        PATH = "./net_NSTN.pt"
        # if loss_list[-1] <= 0.01 and valida_acc >= 0.95:
        #     torch.save(net.state_dict(), PATH)
        #     break

        if early_stopping and loss_list[-2] < loss_list[-1]:  # < 0.05) and (loss_list[-1] <= 0.05):
            if early_epoch == 0: # and valida_acc > 0.9:
                torch.save(net.state_dict(), PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(PATH))
                break
        else:
            early_epoch = 0
        
    # x = range(len(train_acc_list))
    # # y1 = train_acc_list
    # # y2 = valida_acc_list
    # # y3 = train_loss_list
    # y4 = valida_loss_list

    #     # plt.ylim((0.0, 1.0))
    #     # plt.title('扩散速度')  # 折线图标题
    #     # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
    # font1 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 14,
    #          }
    # # 设置横纵坐标的名称以及对应字体格式
    # font2 = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 14,
    #          }
    # plt.xlabel('Epoch', font1)  # x轴标题
    # plt.ylabel('Loss', font2)  # y轴标题

    # # plt.plot(x, y1, color='green', label='train_acc')  # 绘制折线图，添加数据点，设置点的大小
    # # plt.plot(x, y2, color='deepskyblue', label='val_acc')
    # # plt.plot(x, y3, color='gold', label='train_loss')
    # plt.plot(x, y4, color='slategray', label='val_loss')

    # plt.legend(loc='upper right')  # 设置折线名称
    #     # 调整图片中子图的位置
    #     # plt.subplots_adjust(top=0.9, right=0.9)
    # plt.figure()
    # plt.show()  # 显示折线图
    # plt.savefig("acc_loss.jpg")

    '''d2l.set_figsize()
    d2l.plt.figure(figsize=(10, 10))
    train_accuracy = d2l.plt.subplot(221)
    train_accuracy.set_title('train_accuracy')
    d2l.plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train_accuracy')
    # train_acc_plot = np.array(train_acc_plot)
    # for x, y in zip(num_epochs, train_acc_plot):
    #    d2l.plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    test_accuracy = d2l.plt.subplot(222)
    test_accuracy.set_title('valida_accuracy')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('test_accuracy')
    # test_acc_plot = np.array(test_acc_plot)
    # for x, y in zip(num_epochs, test_acc_plot):
    #   d2l.plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    loss_sum = d2l.plt.subplot(223)
    loss_sum.set_title('train_loss')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), train_loss_list, color='red')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train loss')
    # ls_plot = np.array(ls_plot)

    test_loss = d2l.plt.subplot(224)
    test_loss.set_title('valida_loss')
    d2l.plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('valida loss')
    # ls_plot = np.array(ls_plot)

    d2l.plt.show()
    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))'''
