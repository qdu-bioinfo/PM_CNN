import pandas as pd
from tqdm import tqdm
import xlsxwriter
import xlwt

ex_data_path = "../data/"
save_path = '../data/'
name1 = "S_1629.SubjectIBD00"
name2 = "S_1629.SubjectIBD0"
data_arr = []

for i in range(1,10):
    data_path = ex_data_path + name1 + str(i) + "_classification.txt"
    data_arr.append(data_path)
for j in range(10,100):
    data_path = ex_data_path + name2 + str(j) + "_classification.txt"
    data_arr.append(data_path)
print(data_arr)


def txt_trans_xls(filename, res_path, length):
    res_list = []
    for i in range(length):
        result = res_path + str(i) + '.xls'
        res_list.append(result)
        f = open(filename[i])
        x = 0
        y = 0
        xls = xlwt.Workbook()
        sheet = xls.add_sheet('sheet1', cell_overwrite_ok=True)
        while True:
            line = f.readline()
            if not line:
                break
            for i in line.split('\t'):
                item = i.strip()
                sheet.write(x, y, item)
                y += 1
            x += 1
            y = 0
        f.close()
        xls.save(result)
    return res_list


ex_exl_path = txt_trans_xls(data_arr, save_path, 99)


def get_otu_abu(length, ex_path):
    OTU_list = []
    Abundance_list = []
    for i in range(length):
        exl_data = pd.read_excel(ex_path[i])
        OTU_list.append(exl_data['#Database_OTU'].values.tolist())
        Abundance_list.append(exl_data['Abundance'].values.tolist())
    print(OTU_list)
    print(len(OTU_list))
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    all_otu = []
    for i in range(length):
        for j in range(len(OTU_list[i])):
            if OTU_list[i][j] not in all_otu:
                all_otu.append(OTU_list[i][j])

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
          "\n\n\n\n\n")
    print(all_otu)
    print(len(all_otu))
    print("First_otu：", OTU_list[0])
    print("First_Abundance：", Abundance_list[0])

    all_abundance = []
    for x in tqdm(range(len(all_otu))):
        temp = []
        for i in range(length):
            if all_otu[x] in OTU_list[i]:
                in_dex = OTU_list[i].index(all_otu[x])
                temp.append(Abundance_list[i][in_dex])
            else:
                temp.append(0)
        all_abundance.append(temp)

    return all_otu, all_abundance


otu_ex, abundance_ex = get_otu_abu(99, ex_exl_path)


def ex_get():
    workbook = xlsxwriter.Workbook('../data/ex_Abundance_table.xlsx')
    workbook.use_zip64()

    worksheet = workbook.add_worksheet('sheet1')

    for i in range(len(otu_ex)):
        worksheet.write(i, 0, otu_ex[i])
    for i in tqdm(range(len(otu_ex))):
        data = abundance_ex[i]
        for j in range(99):
            worksheet.write(i, j + 1, data[j])

    workbook.close()
    print("successfully！")


ex_get()
