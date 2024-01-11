import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Distance transformation and hierarchical clustering")
    parser.add_argument("--input", "-i", help="Input cophenetic distance matrix", required=True)
    parser.add_argument("--output", "-o", help="Output hierarchical clustering results", required=True)
    args = parser.parse_args()
    input_file = args.input
    base_path, file_name = os.path.split(input_file)
    out_file = os.path.join(base_path, file_name.replace(".csv", "_tran.csv"))

    def drop_row_col(infile, outfile):
        data = pd.read_csv(infile, header=None, skiprows=[0])
        data = data.iloc[:, 1:]
        data.to_csv(outfile, index=False, header=None)

    def zero_one(infile):
        data = pd.read_csv(infile, header=None)
        max_value = data.max().max()
        print("Initial maximum value:", max_value)
        min_value = np.exp(-(max_value*max_value)/0.5)
        print("Transformed minimum value:", min_value)
        min_val = data.min().min()
        print("Initial minimum value:", min_val)
        max_value_transformed = np.exp(-(min_val*min_val)/0.5)
        print("Transformed maximum value:", max_value_transformed)
        print(data)
        df = np.exp(-(data*data)/0.5)
        df = df.astype(np.float32)
        print(df)
        df.to_csv(infile, header=None, index=False)
        return max_value_transformed, min_value

    drop_row_col(input_file, out_file)
    max_value, min_value = zero_one(out_file)

    thresholds = [0.1, 0.01, 0.3, 0.2]
    length = len(thresholds)
    intermediate_file = os.path.join(base_path, file_name.replace(".csv", ".npy"))
    output_file = os.path.join(base_path, file_name.replace(".csv", "_tran.npy"))
    out_file = out_file.replace("\\", "/")
    intermediate_file = intermediate_file.replace("\\", "/")
    intermediate_file = r"" + intermediate_file
    output_file = output_file.replace("\\", "/")

    data1 = pd.read_csv(input_file, header=None)
    np.save(intermediate_file, data1)

    def csv_npy(infile, intermediate_file, outfile):
        data = pd.read_csv(infile, header=None)
        sum_data = data.shape[0]
        print(data)
        np.save(intermediate_file, data)

        data_loaded = np.load(intermediate_file)

        print(data_loaded)
        print("\n")
        return data_loaded, sum_data

    def hierarchical_clustering(distance_matrix):
        def my_dist(p1, p2):
            x = int(p1)
            y = int(p2)
            return 1.0 - distance_matrix[x, y]

        x = list(range(distance_matrix.shape[0]))
        X = np.array(x)
        linked = linkage(np.reshape(X, (len(X), 1)), metric=my_dist, method='single')
        result1 = [[] for _ in range(length)]
        for i, threshold in enumerate(thresholds):
            clusters = fcluster(linked, t=threshold, criterion='distance')
            for j, item in enumerate(clusters):
                result1[i].append(item)
            print(f"Cluster indices at threshold {threshold}:")
            print(result1[i], "\nTotal clusters:", max(result1[i]), "clusters\n")
        # result2 = dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        # indexes = result2.get('ivl')
        # del result2
        del linked
        # index1 = [int(itm) for itm in indexes]
        return result1

    def index_of_duplicates(arr):
        index_dict = {}
        for i, val in enumerate(arr):
            if val in index_dict:
                index_dict[val].append(i)
            else:
                index_dict[val] = [i]
        last_order = [idx for indexes in index_dict.values() for idx in indexes]
        return last_order

    def get_id():
        last_order = [[] for _ in range(length)]
        for i in range(length):
            last_order[i] = index_of_duplicates(result[i])
        return last_order

    data, data_sum = csv_npy(out_file, intermediate_file, output_file)
    result = hierarchical_clustering(data)
    My_order = get_id()

    My_fea_savefile = args.output
    data = pd.read_csv(input_file)
    raw_order = data.iloc[:, 0].tolist()

    def get_fea():
        My_fea = [[] for _ in range(len(My_order_result))]
        for j in range(len(My_order_result)):
            for x in My_order[j]:
                My_fea[j].append(raw_order[x])

        return My_fea

    def save():
        data = pd.DataFrame(My_feature)
        data.to_csv(My_fea_savefile, index=False)
        data1 = pd.read_csv(My_fea_savefile)
        print(data1.head())

    My_order_result = get_id()
    My_feature = get_fea()

    save()


if __name__ == "__main__":
    main()
