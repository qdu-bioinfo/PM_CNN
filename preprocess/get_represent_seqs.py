import pandas as pd

data = pd.read_csv("../data/Gut_Abundance_table_5597.csv")
cols = data.columns.tolist()
print(cols)
print(len(cols))

a = list(map(int, cols))
print(a)
with open('../../dataset/database.txt','r') as file:
    aaa = file.readlines()
    label_list = []
    value_list = []
    for i in range(0,len(aaa),2):
        label_list.append(int(aaa[i][1:len(aaa[i])]))
        value_list.append(aaa[i+1])

with open('../data/Gut_represent_5597.txt','w') as result_file:
    for i in range(len(a)):
        index = label_list.index(a[i])
        result_file.write('>'+str(label_list[index]))
        result_file.write("\n")
        result_file.write(value_list[index])
print("保存完成")
