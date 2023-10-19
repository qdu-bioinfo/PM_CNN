from clustering import My_order
import pandas as pd

print("-----------------------starting----------------------")

input_file = "../data/Oral/oral_1554.csv"
My_fea_savefile = "../data/Oral/Oral_feature.csv"

data = pd.read_csv(input_file)
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

