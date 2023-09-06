import pandas as pd
from tqdm import tqdm
import xlsxwriter
import xlwt

all_list_new = pd.read_csv('../Gut_3113_meta.csv')

All_sampleID = all_list_new['ID'].values.tolist()
print(All_sampleID)
print(len(All_sampleID))

head = '../Gut_new/'
tail = '_classification.txt'
tail_arr = []

filepath = []
for i in range(3113):
    filepath.append(head)
    tail_arr.append(tail)
    filepath[i] = filepath[i] + All_sampleID[i] + tail_arr[i]

print(filepath)

Gut_save_path = '../3113_excel/'
# Gut_path, Gut_len, Gut_start, Gut_end, Gut_sample = Gut_Run()

Gut_len = 3113
Gut_start = 0
Gut_end = 3113


# result = []


def txt_trans_xls(filename, res_path, length, sample_list):
    res_list = []
    for i in range(length):
        result = res_path + sample_list[i] + '.xls'
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
                # item=i.strip().decode('utf8')
                item = i.strip()
                sheet.write(x, y, item)
                y += 1
            x += 1
            y = 0
        f.close()
        xls.save(result)
    return res_list


GUt_exl_path = txt_trans_xls(filepath, Gut_save_path, Gut_len, All_sampleID)
print("GUt_exl_path\n\n\n\n\n\n\n\n\n\n")
print(GUt_exl_path[0], GUt_exl_path[1])


def get_otu_abu(length, ex_path):
    OTU_list = []
    Abundance_list = []
    for i in range(length):
        exl_data = pd.read_excel(ex_path[i])
        # print(pg_exl_data)
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
    a = 0
    b = 0
    print("First_otu：", OTU_list[0])
    print("First_Abundance：", Abundance_list[0])
    for i in range(length):
        num_otu = num_otu + len(OTU_list[i])
        print("id", i + 1, "otu_total：", len(OTU_list[i]))
        num_abundance = num_abundance + len(Abundance_list[i])
        print("id", i + 1, "个Abundance_total：", len(Abundance_list[i]))
    print("sum：", num_otu)
    print("Abundance：", num_abundance)
    print(
        "---------------------------------------------------------------\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        "\n\n\n\n\n\n\n\n\n")
    all_abundance = []
    for x in tqdm(range(len(all_otu))):
        temp = []
        for i in range(length):
            if all_otu[x] in OTU_list[i]:
                # print(OTU_list[i])
                in_dex = OTU_list[i].index(all_otu[x])
                temp.append(Abundance_list[i][in_dex])
            else:
                temp.append(0)
                # break
        all_abundance.append(temp)

    return all_otu, all_abundance


otu_Gut, abundance_Gut = get_otu_abu(Gut_len, GUt_exl_path)


def Gut_get():
    workbook = xlsxwriter.Workbook('../data/Gut_Abundance_table.xlsx')
    workbook.use_zip64()

    worksheet = workbook.add_worksheet('sheet1')

    for i in range(len(otu_Gut)):
        worksheet.write(i, 0, otu_Gut[i])
    for i in tqdm(range(69352)):
        data = abundance_Gut[i]
        for j in range(3113):
            worksheet.write(i, j + 1, data[j])

    workbook.close()
    print("successfully！")


Gut_get()
