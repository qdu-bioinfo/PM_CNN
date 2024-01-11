import pandas as pd
from tqdm import tqdm
import xlsxwriter
import xlwt
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Data preprocess")
    parser.add_argument("--input_sample", "-i", help="The storage path of all samples", required=True)
    parser.add_argument("--meta", "-m", help="Meta information of all samples", required=True)
    parser.add_argument("--output", "-o", help="Output the merged sample abundance table", required=True)
    args = parser.parse_args()

    all_list_new = pd.read_csv(args.meta)
    All_sampleID = all_list_new['SampleID'].values.tolist()

    def get_all_files_in_directory(directory):
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        return all_files

    directory_path = args.input_sample
    all_files_in_directory = get_all_files_in_directory(directory_path)

    for i, file_path in tqdm(enumerate(all_files_in_directory)):
        res_list = []
        result = directory_path + All_sampleID[i] + '.xls'
        res_list.append(result)
        f = open(file_path)
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

    OTU_list = []
    Abundance_list = []

    for i in range(len(all_files_in_directory)):
        exl_data = pd.read_excel(res_list[i])
        OTU_list.append(exl_data['#Database_OTU'].values.tolist())
        Abundance_list.append(exl_data['Abundance'].values.tolist())
    all_otu = []

    for i in range(len(all_files_in_directory)):
        for j in range(len(OTU_list[i])):
            if OTU_list[i][j] not in all_otu:
                all_otu.append(OTU_list[i][j])

    for i in range(len(all_files_in_directory)):
        num_otu = num_otu + len(OTU_list[i])
        print("id", i + 1, "otu_total：", len(OTU_list[i]))
        num_abundance = num_abundance + len(Abundance_list[i])
        print("id", i + 1, "Abundance_total：", len(Abundance_list[i]))

    all_abundance = []
    for x in tqdm(range(len(all_otu))):
        temp = []
        for i in range(len(all_files_in_directory)):
            if all_otu[x] in OTU_list[i]:
                in_dex = OTU_list[i].index(all_otu[x])
                temp.append(Abundance_list[i][in_dex])
            else:
                temp.append(0)
        all_abundance.append(temp)

    workbook = xlsxwriter.Workbook(args.output)
    workbook.use_zip64()
    worksheet = workbook.add_worksheet('sheet1')

    for i in range(len(all_otu)):
        worksheet.write(i, 0, all_otu[i])
    for i in tqdm(range(len(all_otu))):
        data = all_abundance[i]
        for j in range(len(all_files_in_directory)):
            worksheet.write(i, j + 1, data[j])

    workbook.close()

    print("successfully！")


if __name__ == "__main__":
    main()