import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='PM_CNN')

    parser.add_argument('--train_x', type=str, default='../data/Gut/train_data/X_train_5597.csv')

    parser.add_argument('--train_y', type=str, default='../data/Gut/train_data/y_train_5597.csv')

    parser.add_argument('--test_x', type=str, default='../data/Gut/test_data/X_test_5597.csv')

    parser.add_argument('--test_y', type=str, default='../data/Gut/test_data/y_test_5597.csv')

    parser.add_argument('--sequence', type=str, default='../data/Gut/Gut_feature.csv')

    parser.add_argument('--res', type=str, default='../result/')

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--epoch', type=int, default=34)

    parser.add_argument('--learning_rate', type=float, default=5e-3)

    parser.add_argument('--channel', type=int, default=16)

    parser.add_argument('--kernel_size', type=int, default=8)

    parser.add_argument('--strides', type=int, default=4)

    args = parser.parse_args()

    return args