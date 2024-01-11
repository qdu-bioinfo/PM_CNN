import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from config import parse_arguments
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

args = parse_arguments()


def main(args):
    input_file_X_train = args.train_otu_table
    input_file_X_test = args.test_otu_table
    input_file_Y_train = args.meta
    channels = args.channel
    ker_size = args.kernel_size
    strides = args.strides

    def int_str():
        MyFea_df = pd.read_csv(args.clustered_groups)
        row_list = [[] for _ in range(MyFea_df.shape[0])]
        for index, row in MyFea_df.iterrows():
            row_list[index] = list(row)
            row_list[index] = list(map(str, row_list[index]))

        return row_list

    My_list = int_str()

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
            self.fc1 = nn.Linear(11328, 64)
            self.fc2 = nn.Linear(64, 3)

        def conv_block(self, x, conv1, conv2):
            x = F.tanh(conv1(x))
            x = nn.BatchNorm1d(num_features=args.channel)(x)
            x = F.tanh(conv2(x))
            x = nn.BatchNorm1d(num_features=args.channel)(x)
            return x

        def forward(self, x1, x2, x3, x4):
            x1 = x1.reshape(-1, 1, x1.size(1))
            x2 = x2.reshape(-1, 1, x2.size(1))
            x3 = x3.reshape(-1, 1, x3.size(1))
            x4 = x4.reshape(-1, 1, x4.size(1))

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
                if (i + 1) % 2 == 0:
                    print('Epoch [{}/{}]-----------------------Loss: {:.4f}'
                          .format(epoch + 1, EPOCH, loss.item()))
        torch.save(model.state_dict(), args.save_model)

    def PM_CNN_test():
        model = Net()
        model.load_state_dict(torch.load(args.save_model))
        with torch.no_grad():
            outputs = model(test_list[0], test_list[1], test_list[2], test_list[3])
            pred = torch.max(outputs, dim=1)[1]

            print(pred)

    if args.train:

        X_train = pd.read_csv(input_file_X_train)
        Y_train = pd.read_csv(input_file_Y_train)
        Y_train = Y_train.iloc[:, 1].values

        encoder = LabelEncoder()
        Y_train = encoder.fit_transform(Y_train.ravel())
        y_train = torch.LongTensor(Y_train)

        x1_train = X_train[My_list[0]]
        x2_train = X_train[My_list[1]]
        x3_train = X_train[My_list[2]]
        x4_train = X_train[My_list[3]]

        train_list = [x1_train, x2_train, x3_train, x4_train]

        for i in range(len(train_list)):
            train_list[i] = np.array(train_list[i], dtype=np.float32)
            train_list[i] = torch.FloatTensor(train_list[i])

        train_dataset = MyDataset(train_list[0], train_list[1], train_list[2], train_list[3], y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        PM_CNN_train()

    elif args.test:

        X_test = pd.read_csv(input_file_X_test)

        x1_test = X_test[My_list[0]]
        x2_test = X_test[My_list[1]]
        x3_test = X_test[My_list[2]]
        x4_test = X_test[My_list[3]]

        test_list = [x1_test, x2_test, x3_test, x4_test]

        for i in range(len(test_list)):
            test_list[i] = np.array(test_list[i], dtype=np.float32)
            test_list[i] = torch.FloatTensor(test_list[i])

        PM_CNN_test()


if __name__ == '__main__':
    main(args)
