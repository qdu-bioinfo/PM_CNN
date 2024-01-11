import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='PM_CNN')

    parser.add_argument("--train", action="store_true")

    parser.add_argument("--test", action="store_true")

    parser.add_argument('--train_otu_table', type=str, default='../example/train_data/example_abundance_table.csv')

    parser.add_argument('--meta', type=str, default='../example/train_data/meta.csv')

    parser.add_argument('--test_otu_table', type=str, default='../example/test_data/example_test.csv')

    parser.add_argument('--save_model', type=str, default='./PM-CNN_model.pth')

    parser.add_argument('--clustered_groups', type=str, default='../example/output_file.csv')

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--learning_rate', type=float, default=5e-3)

    parser.add_argument('--channel', type=int, default=16)

    parser.add_argument('--kernel_size', type=int, default=8)

    parser.add_argument('--strides', type=int, default=4)

    args = parser.parse_args()

    return args