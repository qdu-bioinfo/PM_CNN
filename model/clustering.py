import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


input_file = "../data/Gut_cd_5597.csv"
out_file = "../data/Gut_cd_5597_tran.csv"


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


# drop_row_col(input_file, out_file)
# max_value, min_value = zero_one(out_file)

print("\n-------------------waiting-----------------------\n")
thresholds = [0.05, 0.1, 0.15, 0.5]
length = len(thresholds)
intermediate_file = r"../data/Gut_cd_5597_tran.npy"
output_file = "../data/Gut_cd_5597_tran.npy"


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
    result1 = [[] for _ in range(length)]
    result_best = fcluster(linked, t=0.51, criterion='distance')
    for i, threshold in enumerate(thresholds):
        clusters = fcluster(linked, t=threshold, criterion='distance')
        for j, item in enumerate(clusters):
            result1[i].append(item)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)

    return result1, result_best


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


def get_id():
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
data = pd.read_csv("../data/Gut_cd_5597.csv")
raw_order = data.iloc[:, 0].tolist()
My_fea = []
for x in result_index:
    My_fea.append(raw_order[x])
# print(result_best)
data = pd.DataFrame(My_fea)
data.to_csv("../data/05_fea.csv", index=False)
My_order = get_id()
# prin()
# plt.show()
# plt.savefig('../data/Gut_clustering_result.png')

print("\n------------------------over---------------------------\n")
