import csv
fo = open("result/country_depth_2.txt", "r+")
data = fo.read()
data = data.replace("'", "")
data = data.split("\n")
insight_file = open('result/country_2.csv', 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(insight_file)
csv_writer.writerow(['Subspace', 'Breakdown', 'Ce', 'Insight_type', 'Imp', 'Sig', 'Score', 'Key', 'Value'])
for str in data:
    str = str[1:-1]
    str = str.split(", ")
    csv_list = []
    i = 0
    while i != len(str):
        if str[i].startswith("Ce"):
            ce_str = str[i].split(":")[1]
            i += 1
            while not str[i].startswith("Type"):
                ce_str += ", " + str[i]
                i += 1
            csv_list.append(ce_str)
        elif str[i].startswith("Key"):
            key_str = str[i].split(":")[1]
            i += 1
            while not str[i].startswith("Value"):
                key_str += ", " + str[i]
                i += 1
            csv_list.append(key_str)
        elif str[i].startswith("Value"):
            value_str = str[i].split(":")[1]
            i += 1
            while i != len(str):
                value_str += ", " + str[i]
                i += 1
            csv_list.append(value_str)
        elif str[i].startswith("Subspace"):
            subspace_str = str[i][9:]
            i += 1
            while not str[i].startswith("Breakdown"):
                subspace_str += ", " + str[i]
                i += 1
            csv_list.append(subspace_str)
        else:
            print(str[i])
            csv_list.append(str[i].split(":")[1])
            i += 1
    print(csv_list)
    csv_writer.writerow(csv_list)
print("读取的字符串是 : ", data)
# 关闭打开的文件
fo.close()