import pandas as pd

data = pd.read_csv("../data/del_ex_Abundance_table.csv", header=None)
print(data)
data.columns = data.iloc[1]
cols = data.columns.tolist()
print(cols)
print(len(cols))

col_name = list(map(int, cols))
with open('../../../database/database13.txt','r') as file:
    lin = file.readlines()
    label_list = []
    value_list = []
    for i in range(0,len(lin),2):
        label_list.append(int(lin[i][1:len(lin[i])]))
        value_list.append(lin[i+1])

with open('../data/ex_represent.txt','w') as result_file:
    for i in range(len(col_name)):
        index = label_list.index(col_name[i])
        result_file.write('>'+str(label_list[index]))
        result_file.write("\n")
        result_file.write(value_list[index])
print("ok")
