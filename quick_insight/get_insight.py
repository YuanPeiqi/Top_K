import csv
import pandas as pd

quick_insight_file = open('quick_insight.txt')
quick_insight_list = quick_insight_file.read().split('\n')
result_file = open('seed_insight.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(result_file)
csv_writer.writerow(['Subspace', 'Breakdown', 'Measure', 'Insight_Type', 'Values'])
for i in range(len(quick_insight_list)):
    record = {}
    for item in quick_insight_list[i].split('@'):
        record[item.split('%')[0]] = item.split('%')[1]
    data_frame = pd.read_csv('quick_insight_data/' + str(i) + '.csv')
    data_dict = data_frame.to_dict(orient="records")
    record['Values'] = [(item[list(item.keys())[0]],item[list(item.keys())[1]]) for item in data_dict]
    csv_writer.writerow([record['Subspace'], record['Breakdown'], record['Measure'], record['Insight_Type'], record['Values']])
