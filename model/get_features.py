from clustering import get_id
import pandas as pd

print("-----------------------starting----------------------")

input_file = "../data/Gut_cd_5597.csv"
My_fea_savefile = "../data/My_cd_feature.csv"

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


My_order = get_id()
My_feature = get_fea()
save()
print("-----------------------over----------------------")

