import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os
sys.path.append(os.pardir)
from global_module import network, train
from global_module.generate_pic import aa_and_each_accuracy, sampling,load_dataset, generate_png, generate_iter
from global_module.Utils import fdssc_model, record, extract_samll_cubic
import tsne
path = '../train'
############ConfusionMatrix_map#########
# class ConfusionMatrix(object):
#     """
#     Note that if the displayed images are incomplete, it is a matplotlib version problem.
#     This routine uses matplotlib-3.2.1 (windows and ubuntu) to draw normally
#     Requires additional installation of the prettytable library
#     """
#     def __init__(self, num_classes: int, labels: list):
#         self.matrix = np.zeros((num_classes, num_classes))
#         self.num_classes = num_classes
#         self.labels = labels
#
#     def update(self, preds, labels):
#         for p, t in zip(preds, labels):
#             self.matrix[p, t] += 1
#
#     def summary(self):
#         # calculate accuracy
#         sum_TP = 0
#         for i in range(self.num_classes):
#             sum_TP += self.matrix[i, i]
#         acc = sum_TP / np.sum(self.matrix)
#         print("the model accuracy is ", acc)
#
#         # precision, recall, specificity
#         table = PrettyTable()
#         table.field_names = ["", "Precision", "Recall", "Specificity"]
#         for i in range(self.num_classes):
#             TP = self.matrix[i, i]
#             FP = np.sum(self.matrix[i, :]) - TP
#             FN = np.sum(self.matrix[:, i]) - TP
#             TN = np.sum(self.matrix) - TP - FP - FN
#             Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
#             Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
#             Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
#             table.add_row([self.labels[i], Precision, Recall, Specificity])
#         print(table)
#
#     def plot(self):
#         matrix = self.matrix
#         print(matrix)
#         plt.figure(figsize=(8, 8))
#         # sns.set(font_scale=0.8)
#         plt.imshow(matrix, cmap=plt.cm.Blues)
#
#
#         plt.xticks(range(self.num_classes), self.labels, rotation=45)
#         # 设置y轴坐标label
#         plt.yticks(range(self.num_classes), self.labels)
#         # 显示colorbar
#         plt.colorbar()
#         plt.xlabel('True Labels',fontsize = 14)
#         plt.ylabel('Predicted Labels',fontsize = 14)
#         plt.title('Confusion matrix',fontsize = 16)
#
#
#         thresh = matrix.max() / 2
#         for x in range(self.num_classes):
#             for y in range(self.num_classes):
#                 # 注意这里的matrix[y, x]不是matrix[x, y]
#                 info = int(matrix[y, x])
#                 plt.text(x, y, info,
#                          verticalalignment='center',
#                          horizontalalignment='center',
#                          color="white" if info > thresh else "black")
#         plt.tight_layout()
#
#         plt.savefig(os.path.join(path, Dataset + ".jpg"))
#         plt.show()
# from sklearn.decomposition import PCA
# def apply_PCA(data, num_components=32):
#     new_data = data.reshape( -1, data.shape[2])
#     pca = PCA(n_components=num_components, whiten=True)
#     new_data = pca.fit_transform(new_data)
#     new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
#     return new_data, pca

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340]
# #seeds = [1331, 1332, 1333, 1334, 1335]
seeds = [1331, 1331, 1331, 1331, 1331, 1331, 1331, 1331, 1331, 1331]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')
global Dataset  # UP,IN,KSC
dataset = input('Please input the name of Dataset(IN, UP, BS, SV, PC or KSC):')
Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(Dataset)
print(data_hsi.shape)
# data_hsi, pca = apply_PCA(data_hsi, num_components=100)
print(data_hsi.shape)

image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 1
PATCH_LENGTH =4
# number of training samples per class
#lr, num_epochs, batch_size = 0.01, 400, 64
lr, num_epochs, batch_size = 0.001, 400,64
#lr, num_epochs, batch_size = 0.0001, 400,64
#lr, num_epochs, batch_size = 0.0050, 400, 64
#lr, num_epochs, batch_size = 0.0005, 400, 64
# net = network.DBDA_network_drop(BAND, CLASSES_NUM)
# net = network.DBDA_network_PReLU(BAND, CLASSES_NUM)
# net = network.DBMA_network(BAND, CLASSES_NUM)
# optimizer = optim.Adam(net.parameters(), lr=lr) #, weight_decay=0.0001)
loss = torch.nn.CrossEntropyLoss()

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

# a=(1,2,3)
# b=(2,3,4)
# c=(0,8,9,5,6)
# print('a',[ att(a,b) for att in c])
KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)
# from torchsummary import summary
for index_iter in range(ITER):
    print('iter:', index_iter)
    net = network.CDC_MDAA(BAND, CLASSES_NUM,2*PATCH_LENGTH+1,2*PATCH_LENGTH+1)
    net = net.to(device)
    # summary(net, (1, 9, 9, BAND))
    optimizer = optim.Adam(net.parameters(), lr=lr,amsgrad=False ) #, weight_decay=0.0001)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)
    SAVE_PATH3 = net.name + '.pth'
    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')

    train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)

    tic1 = time.perf_counter()

    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    best_acc = 0
    # trainloss_txt ="E:\\新建文件夹\\ocean\\train\\intrainLOSS.txt"
    for epoch in range(num_epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        # 原来耐心值5
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 11, eta_min=0.0, last_epoch=-1)
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
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
                net.eval()
                y_hat = net(X)
                l = loss(y_hat, y.long())
                acc_sum += (y_hat.argmax(dim=1) == y.long()).sum().cpu().item()
                test_l_sum += l
                test_num += 1
                net.train()
                n += y.shape[0]
        new_acc = acc_sum / n

        valida_acc = new_acc
        valida_loss = test_l_sum

        #  valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        loss_list.append(valida_loss)
        if valida_acc > best_acc:
            torch.save(net.state_dict(), SAVE_PATH3)
            best_acc = valida_acc

        train_loss_list.append(train_l_sum)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(best_acc)
        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
              % (
              epoch + 1, train_l_sum / batch_count, train_acc_sum / n, valida_loss, best_acc, time.time() - time_epoch))
       
        # outn="%s:enpoch %d,train loss:%.3f,valida loss %.6f, valida acc %.3f"%(epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
        # valida_loss, best_acc)
        # # outp='epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f'% (epoch + 1,
        # #                                 train_l_sum / batch_count, train_acc_sum / n, valida_loss, best_acc)
        # # #
        # # with open(trainloss_txt,"a+") as f:
        #     f.write(outp+'\n')
        #     f.close
       
        early_num=20
        early_stopping = False
        if early_stopping and loss_list[-2] < loss_list[-1]:  # < 0.05) and (loss_list[-1] <= 0.05):
            if early_epoch == 0: # and valida_acc > 0.9:
                torch.save(net.state_dict(), SAVE_PATH3)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(SAVE_PATH3))
                break

        else:
            early_epoch = 0
    #train.train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs)
    toc1 = time.perf_counter()
    #ConfusionMatrix_Label
    # labelsIN = ['Alfalfa', 'CORN–N','Corn–m','Corn','Grass–p','Grass–t','Grass–p–m','Hay-w','Oats','Soybean-n','Soybean-m','Soybean-c','Wheat','Woods','Buildings-G-T','Stone-S-T']
    # labelsSV = ['weeds_1', 'weeds_2', 'Fallow', 'Fallow-r-p', 'Fallow-s', 'Stubble', 'Celery', 'Grapes-u', 'Soil-v-y-d', 'C-s-g-weeds',
    #            'L-r-4wk', 'L-r-5wk', 'L-r-6wk', 'L-r-7wk', 'VIN-yard-u', 'VIN-yard-v-t']
    # labelsUP = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted m s', 'B Soil', 'Bitumen', 'S-B Bricks', 'Shadows']
    # labelsKSC = ['Scrub', 'W swamp', 'CP hammock', 'Slash pine', 'Oak/Broadleaf', 'Hardwood', 'Grass-p-m', 'G marsh', 'Sp marsh',
    #             'C marsh',
    #             'Sa marsh', 'Mud flats', 'Water']
    # confusion = ConfusionMatrix(num_classes=CLASSES_NUM, labels=labelsIN)

    #测试
    pred_test_fdssc = []
    tic2 = time.perf_counter()
    net.load_state_dict(torch.load(SAVE_PATH3))
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            net.eval()
            y_hat = net(X)
            pred_test_fdssc.extend(np.array(net(X).cpu().argmax(axis=1)))
    toc2 = time.perf_counter()
    #tsne

    # from scipy.io import savemat
    # path = "../m.mat"
    # data = {}
    # data.update({'num': pred_test_fdssc})
    # savemat(path, data)
    # # collections.Counter(pred_test_fdssc)
    # print(len(pred_test_fdssc))
    # gt_test = gt[test_indices] - 1
    # import scipy.io as sio
    # out = sio.loadmat('../m.mat')
    # output = out['num']
    # print('1223', output.shape)
    # tsne.tsne(output, gt_test[:-VAL_SIZE])

    collections.Counter(pred_test_fdssc)
    gt_test = gt[test_indices] - 1
    # confusion.update(pred_test_fdssc, gt_test[:-VAL_SIZE])
    # confusion.plot()
    # confusion.summary()


    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test[:-VAL_SIZE])
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test[:-VAL_SIZE])

    torch.save(net.state_dict(), "./net/" + str(round(overall_acc_fdssc, 3)) + '.pt')
    KAPPA.append(kappa)
    OA.append(overall_acc_fdssc)
    AA.append(average_acc_fdssc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc_fdssc

print("--------" + net.name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     'IN/' + net.name + day_str + '_' + Dataset + 'split：' + str(VALIDATION_SPLIT) + 'lr：' + str(lr) + '.txt')
# print(len(total_indices))
generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices)
