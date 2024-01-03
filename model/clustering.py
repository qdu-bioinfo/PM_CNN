import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def main():
    parser = argparse.ArgumentParser(description="Distance transformation and hierarchical clustering")
    parser.add_argument("--input_csv", "-i", help="Input original distance matrix csv file", required=True)
    parser.add_argument("--tran", "-t", help="Get distance converted csv file", required=True)
    parser.add_argument("--npy", "-n", help="Convert the converted matrix csv to npy format", required=True)
    parser.add_argument("--output", "-o", help="Hierarchical clustering result path", required=True)
    args = parser.parse_args()

    input_file = args.input
    out_file = args.tran

    def drop_row_col(infile1, outfile1):
        data = pd.read_csv(infile1, header=None, skiprows=[0])
        data = data.iloc[:, 1:]
        data.to_csv(outfile1, index=False, header=None)

    def zero_one(infile2):
        data = pd.read_csv(infile2, header=None)
        max_data = data.max().max()
        min_value1 = np.exp(-(max_data*max_data)/0.5)
        min_data = data.min().min()
        max_value1 = np.exp(-(min_data*min_data)/0.5)
        # print(data)
        df = np.exp(-(data*data)/0.5)
        df = df.astype(np.float32)
        print(df)
        df.to_csv(infile2, header=None, index=False)
        return max_value1, min_value1

    drop_row_col(input_file, out_file)
    max_value, min_value = zero_one(out_file)

    print("\n-------------------waiting-----------------------\n")
    thresholds = [0.38, 0.45, 0.3, 0.05]
    length = len(thresholds)
    intermediate_file = args.npy
    output_file = args.npy

    def csv_npy(infile1, intermediate_file1, outfile1):
        data = pd.read_csv(infile1, header=None)
        sum_data = data.shape[0]
        np.save(intermediate_file1, data)
        result = np.load(outfile1)
        print("\n")
        return result, sum_data

    def hierarchical_clustering(distance_matrix):
        def my_dist(p1, p2):
            x = int(p1)
            y = int(p2)
            return 1.0 - distance_matrix[x, y]

        x = list(range(distance_matrix.shape[0]))
        X = np.array(x)
        linked = linkage(np.reshape(X, (len(X), 1)), metric=my_dist, method='single')
        result = [[] for _ in range(length)]
        for i, threshold in enumerate(thresholds):
            clusters = fcluster(linked, t=threshold, criterion='distance')
            for j, item in enumerate(clusters):
                result[i].append(item)

        thres = np.linspace(0.05, 0.5, num=10)
        best_result = None
        best_silhouette = -1.0

        for thres in thres:
            cluster_best = fcluster(linked, t=thres, criterion='distance')
            silhouette = silhouette_score(distance_matrix, cluster_best)
            if silhouette > best_silhouette:
                best_result = cluster_best
                best_silhouette = silhouette

        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)

        return result, best_result

    def index_of_duplicates(arr):
        index_dict = {}
        for i, val in enumerate(arr):
            if val in index_dict:
                index_dict[val].append(i)
            else:
                index_dict[val] = [i]
        last_order = []
        for indexes in index_dict.values():
            last_order += indexes
        return last_order

    def get_id(result):
        last_order = [[] for _ in range(length)]
        for i in range(length):
            last_order[i] = index_of_duplicates(result[i])
        return last_order

    def prin():
        for i in range(len(My_order)):
            print("my_data", i + 1, "result：\n", My_order[i])

    data, data_sum = csv_npy(out_file, intermediate_file, output_file)
    result, result_best = hierarchical_clustering(data)
    result_index = index_of_duplicates(result_best)

    data = pd.read_csv(args.input)
    raw_order = data.iloc[:, 0].tolist()
    My_fea = []
    for x in result_index:
        My_fea.append(raw_order[x])
    data = pd.DataFrame(My_fea)
    My_order = get_id(result)
    prin()
    # plt.savefig(args.output)

    # My_fea_savefile = "../data/Oral/Oral_feature.csv"

    My_fea_savefile = args.output
    data = pd.read_csv(args.input)
    raw_order = data.iloc[:, 0].tolist()

    def get_fea():
        My_fea = [[] for _ in range(len(My_order))]
        for j in range(len(My_order)):
            for x in My_order[j]:
                My_fea[j].append(raw_order[x])

        return My_fea

    def save():
        data = pd.DataFrame(My_feature)
        data.to_csv(My_fea_savefile, index=False)
        result = pd.read_csv(My_fea_savefile)
        print(result.head())

    My_feature = get_fea()
    save()
    print("-----------------------over----------------------")


if __name__ == "__main__":
    main()