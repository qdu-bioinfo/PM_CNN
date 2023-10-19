import pandas as pd

data_path = '../data/ex_Abundance_table_T.csv'
# 读取Excel文件
df = pd.read_excel('../data/ex_Abundance_table.xlsx')
df_transposed = df.transpose()
df_transposed.to_csv(data_path, index=False)

df = pd.read_csv(data_path)
print(df)
zero_ratio = (df == 0).sum()/len(df)
col_to_keep = zero_ratio[zero_ratio <= 0.9].index
df = df[col_to_keep]
df.to_csv("../data/del_ex_Abundance_table.csv", index=False)
print(df)