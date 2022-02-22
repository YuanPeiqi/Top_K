import copy
import math
import queue
import time
from math import e

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

# 两种insight评价模式
insight_type = ['Point', 'Shape']
measures = []
current_measure = ''
time_dimension = ''
# datasets = {}
# 处理csv为df
data_frame = None
# data_frame = data_frame.loc[(data_frame != 0).any(axis=1)].dropna()
# data_frame[time_dimension] = data_frame[time_dimension].apply(lambda date: date.split('/')[0])
# car_sales_1 = data_frame.to_dict(orient="records")
# datasets['car_sales_1'] = data_frame
head = []
domain = {}
max_heap = queue.PriorityQueue()
sum_all = 0
k = 0
sibling_cube = {}


class InsightNode:
    def __init__(self, subspace, breakdown, Ce, type, imp, sig, score, result_set):
        self.subspace = subspace
        self.breakdown = breakdown
        self.Ce = Ce
        self.type = type
        self.imp = imp
        self.sig = sig
        self.score = score
        self.result_set = result_set

    def __lt__(self, other):
        return other.score > self.score

    def __str__(self):
        return "subspace:{}, breakdown:{}, Ce:{}, T:{}, score:{}".format(self.subspace, self.breakdown, self.Ce,
                                                                         self.type, self.score)


def SUM(subspace):
    breakdown = None
    flag = True
    for item in subspace.keys():
        if subspace[item] == '*':
            flag = False
            breakdown = item
            break
    # 如果没有可变的维度，就返回空list，不继续遍历，即全部都不为*
    if flag:
        return sum_all
    res = 0
    subspace_key = str({label: (label if subspace[label] != '*' else '*') for label in subspace.keys()})
    values_key = str([value for value in subspace.values() if value != '*'])
    for item in sibling_cube[subspace_key][breakdown][values_key]:
        res += item[1]
    # temp_df = data_frame
    # for column in subspace:
    #     if subspace[column] != '*':
    #         temp_df = temp_df.loc[temp_df[column] == subspace[column]]
    # res = temp_df[current_measure].sum()
    return res


def get_list_by_sibling_cube(subspace, Di):
    flag = True
    for item in subspace.values():
        if item == '*':
            flag = False
            break
    # 如果没有可变的维度，就返回空list，不继续遍历，即全部都不为*
    if flag:
        return []
    temp_subspace = {field: '*' if subspace[field] == '*' else field for field in subspace.keys()}
    fixed_value = [value for value in subspace.values()]
    while '*' in fixed_value:
        fixed_value.remove('*')
    return sibling_cube[str(temp_subspace)][Di][str(fixed_value)]


def construct_sibling_cube(df, headers, df_domain):
    def enumerate_cuboids(dimensions):
        temp = []
        N = len(dimensions)
        for i in range(2 ** N - 1, -1, -1):  # 子集的个数，sibling cube不包括空集，在最后会将空集情况直接加入
            item = []
            for idx in range(N):  # 用来判断二进制数的下标为j的位置的数是否为1
                if (i >> idx) % 2:
                    item.append(dimensions[idx])
                else:
                    item.append('*')
            for idx in range(len(item)):
                if item[idx] == '*':
                    temp.append((item, dimensions[idx]))
        return temp

    def dfs_generate_cuboid(current_df, index, current_key, Di):
        if index == len(cuboid[0]):
            value_list = []
            for value in df_domain[Di]:
                a = current_df.loc[current_df[Di] == value]
                Mi = a[current_measure].sum()
                value_list.append((value, Mi))
            # 按照M降序排列,检索key为(ai, bi, ci,...),一个tuple
            value_list = sorted(value_list, key=lambda x: x[1], reverse=True)
            sibling_cube[subspace_key][breakdown_key][str(current_key)] = value_list
        else:
            # 遍历当前subspace中不为*的字段的所有取值
            if cuboid[0][index] != '*':
                for value in df_domain[cuboid[0][index]]:
                    temp_key = copy.deepcopy(current_key)
                    temp_key.append(value)
                    temp_df = current_df.loc[current_df[cuboid[0][index]] == value]
                    dfs_generate_cuboid(temp_df, index + 1, temp_key, Di)
            else:
                dfs_generate_cuboid(current_df, index + 1, current_key, Di)

    # clear sibling_cube for new data
    sibling_cube.clear()
    cuboid_list = enumerate_cuboids(headers)
    for cuboid in cuboid_list:
        print(cuboid)
        subspace_key = str({headers[i]: cuboid[0][i] for i in range(len(cuboid[0]))})
        breakdown_key = cuboid[1]
        if subspace_key not in sibling_cube.keys():
            sibling_cube[subspace_key] = {}
        sibling_cube[subspace_key][breakdown_key] = {}
        dfs_generate_cuboid(df, 0, [], cuboid[1])


# 根据domain的大小进行排序
def ordering_by_domain_size(headers, df_domain):
    sorted_headers = copy.deepcopy(headers)
    return sorted(sorted_headers, key=lambda x: len(df_domain[x]))


# 当前Sibling Group与Ce组合是否合法
def is_valid(S, breakdown, Ce):
    for pair in Ce:
        if breakdown != pair[1] and pair[1] in S.keys() and S[pair[1]] == '*':
            return False
    return True


# 当遍历到最底层时，计算对应的统计量
def calculate(result_set, extractor, subspace):
    type = extractor[0]
    field = extractor[1]
    if type == 'rank':
        result_set = sorted(result_set, key=lambda x: x[1], reverse=True)
        for item in result_set:
            if item[0] == subspace:
                return result_set.index(item) + 1
        return None
    elif type == '%':
        sum_temp = 0
        current_value = 0
        for item in result_set:
            if item[0] == subspace:
                current_value = item[1]
            sum_temp += item[1]
        if sum_temp != 0:
            return round(current_value / sum_temp, 4)
        else:
            return 0
    elif type == 'avg':
        sum_temp = 0
        current_value = 0
        count = 0
        for item in result_set:
            count += 1
            sum_temp += item[1]
            if item[0] == subspace:
                current_value = item[1]
        if count != 0:
            avg = sum_temp / count
            return current_value - avg
        else:
            return None
    else:
        result_set = sorted(result_set, key=lambda x: x[0][field], reverse=False)
        for item in result_set:
            if item[0] == subspace:
                idx = result_set.index(item)
                if idx != 0:
                    return result_set[idx][1] - result_set[idx - 1][1]
                else:
                    return None


def recur_extract(subspace, level, Ce):
    if level > 0:
        result_set = []
        breakdown = Ce[level][1]
        for v in domain[breakdown]:
            temp_subspace = copy.deepcopy(subspace)
            temp_subspace[breakdown] = v
            measure_v = recur_extract(temp_subspace, level - 1, Ce)
            result_set.append((temp_subspace, measure_v))
        measure = calculate([i for i in result_set if i[1]], Ce[level], subspace)
        return measure
    else:
        measure = SUM(subspace)
        return measure


def extract(subspace, breakdown, Ce):
    result_set = []
    for v in domain[breakdown]:
        temp_subspace = copy.deepcopy(subspace)
        temp_subspace[breakdown] = v
        measure = recur_extract(temp_subspace, len(Ce) - 1, Ce)
        result_set.append((temp_subspace, measure))
    # for item in result_set:
    #     print(item)
    return result_set


def get_impact(subspace, breakdown):
    imp_sum = 0
    for v in domain[breakdown]:
        temp_subspace = copy.deepcopy(subspace)
        temp_subspace[breakdown] = v
        temp_df = data_frame
        for column in temp_subspace:
            if temp_subspace[column] != '*':
                temp_df = temp_df.loc[temp_df[column] == temp_subspace[column]]
        imp_sum += temp_df[current_measure].sum()
    return imp_sum / sum_all


def get_sig(result_set, T):
    sig = 0
    result_copy = [item[1] for item in result_set if item[1] is not None]
    # 如果当前Insight没有结果集或者结果集过小，则sig为0
    if len(result_copy) <= 1:
        return 0
    if T == 'Point':
        # 结果集只有两项，则计算差值绝对值与第二项绝对值的比值，映射到0-1
        if len(result_copy) == 2:
            return 1 - (e ** abs(result_copy[0] - result_copy[1]) / abs(result_copy[1] / 3))
        temp_set = sorted(result_copy, reverse=True)
        x_max = float(temp_set[0])
        x_min = float(temp_set[len(temp_set) - 1])
        if x_min <= 0:
            # 将数据全部调整为正数
            for i in range(len(temp_set)):
                temp_set[i] += 1.5 * abs(x_min) + 1
        y = []
        for i in range(1, len(temp_set)):
            y.append(temp_set[i])
        y = np.array(y)
        x = np.array([i + 2 for i in range(len(y))])
        # 利用幂函数性质进行线性拟合
        linear_coefficient = np.polyfit(np.log(x), np.log(y), 1)
        linear_fitted_val = np.polyval(linear_coefficient, np.log(x))
        # 利用拟合的幂函数求得真实拟合值
        real_fitted_val = np.exp(linear_fitted_val)
        # 获得残差
        residuals = np.subtract(real_fitted_val, y)
        residual_0 = abs(math.exp(linear_coefficient[0] * math.log(1) + linear_coefficient[1]) - x_max)
        # 进行正态分析
        mean = np.mean(residuals)  # 计算均值
        std = np.std(residuals)  # 计算标准差
        sig = 2 * get_probability_normal_distribution(residual_0, mean, std) - 1
    elif T == 'Shape':
        y = np.array(result_copy)
        x = np.array([i + 1 for i in range(len(y))])
        # 线性拟合并求得拟合值
        linear_coefficient = np.polyfit(x, y, 1)
        linear_fitted_val = np.polyval(linear_coefficient, x)
        # 求得r2
        r2 = r2_score(y, linear_fitted_val, multioutput='raw_values')
        sig = r2 * (2 * get_probability_normal_distribution(linear_coefficient[0], mean=0.2, std=2) - 1)
    return sig


# 用于计算正态分布的概率
def get_probability_normal_distribution(x, mean, std):
    # 如果std为0，则说明没有意义，sig = 0
    if std == 0:
        return 0.5
    z = (x - mean) / std
    probability = stats.norm.cdf(z)
    return probability


def enumerate_insight(subspace, breakdown, Ce):
    # enumerate all valid SG for current Ce
    ubk = 0
    # 如果size比k小，则ubk为0，即无限制，若size为k则ubk取第k个insight的score
    if max_heap.qsize() >= k:
        k_insight = max_heap.get()
        max_heap.put(k_insight)
        ubk = k_insight.score
    if is_valid(subspace, breakdown, Ce):
        imp = get_impact(subspace, breakdown)
        if ubk <= imp:
            result_set = extract(subspace, breakdown, Ce)
            for T in insight_type:
                sig = get_sig(result_set, T)
                score = imp * sig
                if score > ubk:
                    max_heap.put(InsightNode(subspace, breakdown, Ce, T, imp, sig, score, result_set))
                    # 将超出k的insight删除，只保留前k个，这样可以保证优先队列的排序效率
                    while max_heap.qsize() > k:
                        max_heap.get()
    sorted_list = get_list_by_sibling_cube(subspace, breakdown)
    for value in sorted_list:
        temp_subspace = copy.deepcopy(subspace)
        temp_subspace[breakdown] = value[0]
        subspace_sum = value[1]
        for key in temp_subspace.keys():
            if temp_subspace[key] == '*':
                enumerate_insight(temp_subspace, key, Ce)


def dfs_Ce(aggregation, combination, depth, level, Ce, Ces):
    if level == 0:
        Ces.append(Ce)
        return
    if level == depth:
        for agg in aggregation:
            Ce = [agg]
            dfs_Ce(aggregation, combination, depth, level - 1, Ce, Ces)
    else:
        for com in combination:
            temp = [item for item in Ce]
            temp.append(com)
            dfs_Ce(aggregation, combination, depth, level - 1, temp, Ces)


def enumerate_Ce(headers, measure, time_col, depth):
    # enumerate all valid Ce
    extractors = ['rank', '%', 'avg']
    combination = [('prev', time_col)]
    # aggregation = [('SUM', measure), ('COUNT', measure), ('AVERAGE', measure), ('MAX', measure), ('MIN', measure)]
    aggregation = [('SUM', measure)]
    for header in headers:
        for extractor in extractors:
            combination.append((extractor, header))
    Ces = []
    dfs_Ce(aggregation, combination, depth, depth, [], Ces)
    return Ces


def insight(depth, insight_size, file_name, m, t):
    global max_heap
    global k
    global sum_all
    global head
    global domain
    # sibling_cube 存储数据, a hash table
    global sibling_cube
    global measures
    global current_measure
    global time_dimension
    global data_frame

    # 度量字段
    measures = m
    # 时间性字段
    time_dimension = t
    # 处理csv为df
    data_frame = pd.read_csv('dataset/' + file_name + '.csv')
    data_frame = data_frame.loc[(data_frame != 0).any(axis=1)].dropna()
    # data_frame = data_frame.drop('region', axis=1)
    # data_frame = data_frame.drop('country', axis=1)
    data_frame[time_dimension] = data_frame[time_dimension].apply(lambda date: date.split('/')[0])
    data_dict = data_frame.to_dict(orient="records")
    # 列举非度量字段为head，并得到domain
    head = [i for i in data_dict[0].keys() if i not in measures]
    # domains for all columns
    for col in head:
        values = list(data_frame[col].unique())
        domain[col] = values
    print('Domain for ' + file_name + ':' + str(domain))
    for item in domain:
        print(item, ': ', domain[item])
        print('size', len(domain[item]))
    # heap size k
    k = insight_size
    # for every single measure
    for item in measures:
        insight_time = time.perf_counter()
        print("Current Measure: ", item)
        current_measure = item

        # 构建sibling cube
        print('In function construct_sibling_cube for ' + file_name)
        sibling_cube_time = time.perf_counter()
        construct_sibling_cube(data_frame, head, domain)
        print(sibling_cube)
        print('Construct_sibling_cube finished. Time: ' + str(time.perf_counter() - sibling_cube_time) + "s")

        # sum_all计算S的和，因为在impact中是固定的，减少计算次数
        sum_all = data_frame[current_measure].sum()
        # insight maximum heap
        max_heap = queue.PriorityQueue()
        Ce_list = enumerate_Ce(headers=head, measure=current_measure, time_col=time_dimension, depth=depth)
        print('Ce_list for ' + file_name + ': ' + str(Ce_list))

        print('Calculation for ' + file_name)
        # 对head进行sort，按照domain从小到大的顺序遍历每个字段
        ordered_dimensions = ordering_by_domain_size(head, domain)
        for Ce_current in Ce_list:
            for dimension in ordered_dimensions:
                print("Current Ce: " + str(Ce_list.index(Ce_current)))
                print("Current dimension: " + str(ordered_dimensions.index(dimension)))
                subspace = {i: '*' for i in head}
                enumerate_insight(subspace, dimension, Ce_current)
        print("Current insight calculation finished. Time: " + str(time.perf_counter() - insight_time) + "s")

        insight_list = [max_heap.get() for _ in range(k)]
        insight_file = open('result/' + file_name + '_depth_' + str(depth) + '.txt', 'w', encoding='utf-8', newline="")
        res = []
        for i in reversed(range(len(insight_list))):
            temp = [item if item[1] else (item[0], 0) for item in insight_list[i].result_set]
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            insight_file.write('{'
                       'Subspace:\'' + str(insight_list[i].subspace).replace('\'', '') + '\', ' +
                       'Breakdown:\'' + str(insight_list[i].breakdown) + '\', ' +
                       'Ce:\'' + str(insight_list[i].Ce).replace('\'', '') + '\', ' +
                       'Type:\'' + str(insight_list[i].type) + '\', ' +
                       'Sig:' + str(insight_list[i].sig).replace("[", "").replace("]", "")[0:7] + ', ' +
                       'Impact:' + str(insight_list[i].imp).replace("[", "").replace("]", "")[0:7] + ', ' +
                       'Score:' + str(insight_list[i].score).replace("[", "").replace("]", "")[0:7] + ', ' +
                       'Key:' + str([item[0][str(insight_list[i].breakdown)] for item in temp]) + ', ' +
                       'Value:' + str([item[1] if item[1] else 0 for item in temp]) +
                       '}\n')
            res.append('{'
                       'Subspace:\'' + str(insight_list[i].subspace).replace('\'', '') + '\', ' +
                       'Breakdown:\'' + str(insight_list[i].breakdown) + '\', ' +
                       'Ce:\'' + str(insight_list[i].Ce).replace('\'', '') + '\', ' +
                       'Type:\'' + str(insight_list[i].type) + '\', ' +
                       'Sig:' + str(insight_list[i].sig).replace("[", "").replace("]", "")[0:7] + ', ' +
                       'Impact:' + str(insight_list[i].imp).replace("[", "").replace("]", "")[0:7] + ', ' +
                       'Score:' + str(insight_list[i].score).replace("[", "").replace("]", "")[0:7] + ', ' +
                       'Key:' + str([item[0][str(insight_list[i].breakdown)] for item in temp]) + ', ' +
                       'Value:' + str([item[1] if item[1] else 0 for item in temp]) +
                       '}')
        print(str(res).replace('\"', ''))
        insight_file.close()
        # 计算结束，复制一份result_set写入文件
        # insight_file = open('result/' + file_name + 'depth_' + str(depth) + '.txt', 'w', encoding='utf-8', newline="")
        # csv_writer = csv.writer(insight_file)
        # csv_writer.writerow(['Subspace', 'Breakdown', 'Ce', 'Insight_type', 'Imp', 'Sig', 'Score'])
        # while not max_heap.empty() or insight_size > 0:
        #     insight_size -= 1
        #     current_insight = max_heap.get()
        #     csv_writer.writerow(
        #         [current_insight.subspace, current_insight.breakdown, current_insight.Ce, current_insight.type,
        #          current_insight.imp, current_insight.sig, current_insight.score])
        # insight_file.close()
    print("All finished!")


if __name__ == '__main__':
    # car_sales_1
    start_time = time.perf_counter()
    insight(2, 100, 'carSales1', ['Sales'], 'Year')
    time_elapsed = time.perf_counter() - start_time
    print("Total time: " + str(time_elapsed) + "s")
