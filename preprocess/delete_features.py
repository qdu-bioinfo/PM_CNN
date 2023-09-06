import pandas as pd

df = pd.read_csv('../data/Gut_Abundance_table_T.csv')
# print(df.shape)
print(df)
zero_ratio = (df == 0).sum()/len(df)
col_to_keep = zero_ratio[zero_ratio <= 0.9].index
df = df[col_to_keep]
df.to_csv("../data/Gut_Abundance_table_5597.csv", index=False)
print(df)