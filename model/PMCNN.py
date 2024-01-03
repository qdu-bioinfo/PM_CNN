import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from config import parse_arguments
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

args = parse_arguments()
label_num = 3


def main(args):
    input_file_X_train = args.train_x
    input_file_X_test = args.test_x
    input_file_Y_train = args.train_y
    input_file_Y_test = args.test_y
    channels = args.channel
    ker_size = args.kernel_size
    strides = args.strides
    res = args.res

    def int_str():
        MyFea_df = pd.read_csv(args.sequence)
        row_list = [[] for _ in range(MyFea_df.shape[0])]
        for index, row in MyFea_df.iterrows():
            row_list[index] = list(row)
            row_list[index] = list(map(str, row_list[index]))

        return row_list

    My_list = int_str()

    X_train = pd.read_csv(input_file_X_train)
    Y_train = pd.read_csv(input_file_Y_train)
    Y_train = Y_train.iloc[:, 1].values

    X_test = pd.read_csv(input_file_X_test)
    Y_test = pd.read_csv(input_file_Y_test)
    Y_test = Y_test.iloc[:, 1].values

    encoder = LabelEncoder()

    Y_train = encoder.fit_transform(Y_train.ravel())
    y_train = torch.LongTensor(Y_train)
    Y_test = encoder.fit_transform(Y_test.ravel())
    y_test = torch.LongTensor(Y_test)

    x1_train = X_train[My_list[0]]
    x2_train = X_train[My_list[1]]
    x3_train = X_train[My_list[2]]
    x4_train = X_train[My_list[3]]

    x1_test = X_test[My_list[0]]
    x2_test = X_test[My_list[1]]
    x3_test = X_test[My_list[2]]
    x4_test = X_test[My_list[3]]

    train_list = [x1_train, x2_train, x3_train, x4_train]
    test_list = [x1_test, x2_test, x3_test, x4_test]

    def arr_tensor():
        for i in range(len(train_list)):
            train_list[i] = np.array(train_list[i], dtype=np.float32)
            test_list[i] = np.array(test_list[i], dtype=np.float32)
            train_list[i] = torch.FloatTensor(train_list[i])
            test_list[i] = torch.FloatTensor(test_list[i])

        return train_list, test_list

    print("Let's start training the model!\n\n\n\n")
    print("Loading......\n\n")

    class MyDataset(Dataset):
        def __init__(self, x1, x2, x3, x4, y):
            self.x1 = x1
            self.x2 = x2
            self.x3 = x3
            self.x4 = x4
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, index):
            return (self.x1[index], self.x2[index], self.x3[index], self.x4[index]), self.y[index]

    x_train_list, x_test_list = arr_tensor()

    train_dataset = MyDataset(x_train_list[0], x_train_list[1], x_train_list[2], x_train_list[3], y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = MyDataset(x_test_list[0], x_test_list[1], x_test_list[2], x_test_list[3], y_test)
    test_loader = DataLoader(test_dataset, batch_size=477)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1_1 = nn.Conv1d(1, out_channels=channels, kernel_size=ker_size, stride=strides, padding=1)
            self.conv1_2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=ker_size, stride=strides,
                                     padding=1)
            self.conv2_1 = nn.Conv1d(1, out_channels=channels, kernel_size=ker_size, stride=strides, padding=1)
            self.conv2_2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=ker_size, stride=strides,
                                     padding=1)
            self.conv3_1 = nn.Conv1d(1, out_channels=channels, kernel_size=ker_size, stride=strides, padding=1)
            self.conv3_2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=ker_size, stride=strides,
                                     padding=1)
            self.conv4_1 = nn.Conv1d(1, out_channels=channels, kernel_size=ker_size, stride=strides, padding=1)
            self.conv4_2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=ker_size, stride=strides,
                                     padding=1)
            self.fc1 = nn.Linear(24576, 64)
            self.fc2 = nn.Linear(64, 3)

        def conv_block(self, x, conv1, conv2):
            x = F.tanh(conv1(x))
            x = nn.BatchNorm1d(num_features=64)(x)
            x = F.tanh(conv2(x))
            x = nn.BatchNorm1d(num_features=64)(x)
            return x

        def forward(self, x1, x2, x3, x4):
            x1 = x1.reshape(-1, 1, 1554)
            x2 = x2.reshape(-1, 1, 1554)
            x3 = x3.reshape(-1, 1, 1554)
            x4 = x4.reshape(-1, 1, 1554)

            x1 = self.conv_block(x1, self.conv1_1, self.conv1_2)
            x2 = self.conv_block(x2, self.conv2_1, self.conv2_2)
            x3 = self.conv_block(x3, self.conv3_1, self.conv3_2)
            x4 = self.conv_block(x4, self.conv4_1, self.conv4_2)

            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            x3 = x3.view(x3.size(0), -1)
            x4 = x4.view(x4.size(0), -1)
            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = self.fc1(x)
            x = nn.BatchNorm1d(num_features=64)(x)
            x = F.tanh(x)
            x = F.softmax(self.fc2(x))

            return x

    def PM_CNN_train():
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-3)
        EPOCH = args.epoch
        for epoch in range(EPOCH):
            for i, data in enumerate(train_loader):
                inputs, labels = data
                x_train1, x_train2, x_train3, x_train4 = inputs
                y_pred = model(x_train1, x_train2, x_train3, x_train4)
                loss = criterion(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}]-----------------------Loss: {:.4f}'
                          .format(epoch + 1, EPOCH, loss.item()))
        torch.save(model.state_dict(), "./PM_CNN_model.pth")

    def PM_CNN_test():
        model = Net()
        model.load_state_dict(torch.load("./PM_CNN_model.pth"))
        probabilities = []
        true_labels = []
        with torch.no_grad():
            y_pre = []
            y_label = []
            for i, data1 in enumerate(test_loader):
                data2, label2 = data1
                x_test1, x_test2, x_test3, x_test4 = data2
                outputs = model(x_test1, x_test2, x_test3, x_test4)
                probabilities.append(outputs.numpy())
                true_labels.append(label2.numpy())
                pred = torch.max(outputs, dim=1)[1]
                y_pre.extend(pred.tolist())
                y_label.extend(label2.tolist())

            pred_classes = pred
            probabilities = np.concatenate(probabilities)
            true_labels = np.concatenate(true_labels)

            kappa_score = cohen_kappa_score(y_label, y_pre)
            print("Kappa Score: {:.4f}".format(kappa_score))

            accuracy = accuracy_score(y_label, pred_classes)
            f1 = f1_score(y_label, pred_classes, average=None)
            precision = precision_score(y_label, pred_classes, average=None)
            recall = recall_score(y_label, pred_classes, average=None)
            for i in range(len(precision)):
                print(f"Class {i}: Precision={precision[i]}, Recall={recall[i]}, F1={f1[i]}")

        print('Accuracy:', accuracy)

        return probabilities, true_labels

    def draw_ROC_curve(total_label, probabilities, true_labels):
        file_namelist = ['Control', 'Gingivitis', 'Periodontitis']
        plt.figure(figsize=(10, 7), dpi=1600)
        binarized_labels = label_binarize(true_labels, classes=[0, 1, 2])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(total_label):
            fpr[i], tpr[i], _ = roc_curve(binarized_labels[:, i], probabilities[:, i])
            roc_auc[i] = 100 * auc(fpr[i], tpr[i])

        plt.figure()

        for i in range(total_label):
            plt.plot(fpr[i], tpr[i], lw=1, alpha=0.7, label='%s    (AUC=%0.2f%%)' % (file_namelist[i], roc_auc[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right", prop={'size': 8})
        plt.savefig(res + 'result.png')
        # plt.show()

    if args.train:
        PM_CNN_train()
    elif args.test:
        proba, tru_label = PM_CNN_test()
        draw_ROC_curve(label_num, proba, tru_label)


if __name__ == '__main__':
    main(args)