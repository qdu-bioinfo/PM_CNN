import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Process Abundance Table Data")
    parser.add_argument("--input", "-i", help="Input CSV file path", required=True)
    parser.add_argument("--output", "-o", help="Output TXT file path", required=True)
    args = parser.parse_args()

    # Read input CSV file
    data = pd.read_csv(args.input, header=None)
    print(data)
    
    # Set column names
    data.columns = data.iloc[1]
    cols = data.columns.tolist()
    print(cols)
    print(len(cols))

    col_name = list(map(int, cols))
    with open('../../../database/database13.txt', 'r') as file:
        lin = file.readlines()
        label_list = []
        value_list = []
        for i in range(0, len(lin), 2):
            label_list.append(int(lin[i][1:len(lin[i])]))
            value_list.append(lin[i + 1])

    with open(args.output, 'w') as result_file:
        for i in range(len(col_name)):
            index = label_list.index(col_name[i])
            result_file.write('>' + str(label_list[index]))
            result_file.write("\n")
            result_file.write(value_list[index])
    print("ok")

if __name__ == "__main__":
    main()
