import csv
import math
import os
import time

import pymysql
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


# insight_types = ['Attribution', 'Outstanding No.1', 'Evenness', 'Trend', 'Outstanding No.2', 'Outstanding Last', 'Change Point', 'Outlier']
insight_types = ['Attribution', 'Outstanding No.1', 'Evenness', 'Trend', 'Outstanding No.2']
host = 'localhost'
port = 3306
user = 'root'
password = 'ypq001207'
db = 'insight'
charset = 'utf8'


def get_conn():
    conn = pymysql.connect(host=host, port=port, user=user, passwd=password, db=db, charset=charset)
    return conn


def select(cur, subspace, breakdown, exd_dim, measure, measure_col, table_name):
    sql = 'select ' + exd_dim + ',' + breakdown + ',' + measure + ' from ' + table_name + ' where true '
    for item in subspace.keys():
        if subspace[item] != '*' and item != exd_dim:
            sql += 'and ' + str(item) + " = '" + str(subspace[item]) + "'"
    sql += ' group by ' + exd_dim + ', ' + breakdown + ' order by ' + exd_dim + ', ' + breakdown
    cur.execute(sql)
    res = list(cur.fetchall())
    sql = 'select sum(' + measure_col + ') from ' + table_name
    cur.execute(sql)
    total = float(cur.fetchone()[0])
    return res, total


def construct_HDS(subspace, breakdown, exd_dim, measure, measure_col, table_name):
    conn = get_conn()
    cur = conn.cursor()
    res, total = select(cur, subspace, breakdown, exd_dim, measure, measure_col, table_name)
    data = {}
    for item in res:
        if item[0] not in data.keys():
            data[item[0]] = {}
        if item[1] not in data[item[0]].keys():
            data[item[0]][item[1]] = float(item[2])
    conn.close()
    return data, total


def construct_HDP(HDS, insight_type, total):
    res = []
    for item in HDS.keys():
        flag, highlight, imp, sig, values = dp(HDS[item], insight_type, total)
        if flag:
            res.append((item, insight_type, highlight, imp, sig, imp*sig, values))
    return sorted(res, key=lambda x: x[5], reverse=True)


# 最大一项比例大于平均的三倍, highlight为占比最高的一项, sig = 占比最高的一项 / (占比最高的一项 + 各项占比的平均值)
def is_Attribution(dataset):
    flag, highlight, sig, values = False, None, None, list(dataset.values())
    total = sum(values)
    highlight = max(values) / total
    mean = np.mean(values) / total
    if highlight > 5 * mean:
        flag = True
        sig = highlight / (highlight + mean)
        sorted_keys = sorted(dataset, key=lambda x: dataset[x], reverse=True)
        values = [(item, dataset[item] / total) for item in sorted_keys]
    return flag, highlight, sig, values


def get_probability_normal_distribution(x, mean, std):
    # 如果std为0，则说明没有意义，sig = 0
    if std == 0:
        return 1
    z = (x - mean) / std
    probability = 1 - stats.norm.cdf(z)
    return probability


# 计算方式同Point
def is_Outstanding_No1(dataset):
    flag, highlight, sig, values = False, None, None, list(dataset.values())
    sorted_keys = sorted(dataset, key=lambda a: dataset[a], reverse=True)
    sorted_values = sorted(values, reverse=True)
    maximum, minimum = sorted_values[0], sorted_values[-1]
    if len(sorted_values) <= 2 or minimum < 0:
        return flag, highlight, sig, values
    y = sorted_values[1:]
    y = np.array(y)
    # x为index, 用于拟合
    x = np.array([i + 2 for i in range(len(y))])
    # 利用幂函数性质进行线性拟合
    linear_coefficient = np.polyfit(np.log(x), np.log(y), 1)
    linear_fitted_val = np.polyval(linear_coefficient, np.log(x))
    # 利用拟合的幂函数求得真实拟合值
    real_fitted_val = np.exp(linear_fitted_val)
    # 获得残差
    residuals = np.subtract(real_fitted_val, y)
    residual_0 = maximum - math.exp(linear_coefficient[0] * math.log(1) + linear_coefficient[1])
    # 进行正态分析
    mean = np.mean(residuals)  # 计算均值
    std = np.std(residuals)  # 计算标准差
    # print(residual_0, mean, std ,get_probability_normal_distribution(residual_0, mean, std))
    sig = 1 - get_probability_normal_distribution(residual_0, mean, std)
    if sig > 0.5:
        flag, highlight, sig, values = True, maximum, sig, [(key, dataset[key]) for key in sorted_keys]
    return flag, highlight, sig, values


# 将No1与No2都拿出，将他们都当做Outstanding_No1分别处理一次，将sig取平均
def is_Outstanding_No2(dataset):
    flag, highlight, sig, values = False, None, None, list(dataset.values())
    sorted_keys = sorted(dataset, key=lambda a: dataset[a], reverse=True)
    sorted_values = sorted(values, reverse=True)
    if len(sorted_values) <= 3:
        return flag, highlight, sig, values
    No1, No2, minimum = sorted_values[0], sorted_values[1], sorted_values[-1]
    if minimum < 0:
        return flag, highlight, sig, values
    y = np.array(sorted_values[2:])
    # x为index, 用于拟合
    x = np.array([i + 2 for i in range(len(y))])
    # 利用幂函数性质进行线性拟合
    linear_coefficient = np.polyfit(np.log(x), np.log(y), 1)
    linear_fitted_val = np.polyval(linear_coefficient, np.log(x))
    # 利用拟合的幂函数求得真实拟合值
    real_fitted_val = np.exp(linear_fitted_val)
    # 获得残差
    residuals = np.subtract(real_fitted_val, y)
    residual_1 = No1 - math.exp(linear_coefficient[0] * math.log(1) + linear_coefficient[1])
    residual_2 = No2 - math.exp(linear_coefficient[0] * math.log(1) + linear_coefficient[1])
    mean = np.mean(residuals)  # 计算均值
    std = np.std(residuals)  # 计算标准差
    sig_1 = 1 - get_probability_normal_distribution(residual_1, mean, std)
    sig_2 = 1 - get_probability_normal_distribution(residual_2, mean, std)
    sig = (sig_1 + sig_2) / 2
    if sig > 0.5:
        flag = True
        highlight = (No1, No2)
        sig = sig
        values = [(key, dataset[key]) for key in sorted_keys]
    return flag, highlight, sig, values


# 将所有负值都拿出，取绝对之后做OutStanding_No1
def is_Outstanding_Last(dataset):
    flag, highlight, sig, values = False, None, None, [abs(value) for value in dataset.values() if value < 0]
    sorted_keys = sorted(dataset, key=lambda a: dataset[a], reverse=True)
    sorted_values = sorted(values, reverse=True)
    if len(sorted_values) <= 2:
        # 如果只有一个负值，其余全是正值，也可算作Last
        if len(sorted_values) == 1:
            return True, -sorted_values[0], 0.6, [(key, dataset[key]) for key in sorted_keys]
        return flag, highlight, sig, values
    maximum = sorted_values[0]
    y = np.array(sorted_values[1:])
    x = np.array([i + 2 for i in range(len(y))])
    linear_coefficient = np.polyfit(np.log(x), np.log(y), 1)
    linear_fitted_val = np.polyval(linear_coefficient, np.log(x))
    real_fitted_val = np.exp(linear_fitted_val)
    residuals = np.subtract(real_fitted_val, y)
    residual_0 = maximum - math.exp(linear_coefficient[0] * math.log(1) + linear_coefficient[1])
    mean = np.mean(residuals)
    std = np.std(residuals)
    sig = 1 - get_probability_normal_distribution(residual_0, mean, std)
    if sig > 0.5:
        flag = True
        highlight = -maximum
        sig = sig
        values = [(key, dataset[key]) for key in sorted_keys]
    return flag, highlight, sig, values


def is_Change_Point(dataset):
    flag, highlight, sig, values = False, None, None, None
    return flag, highlight, sig, values


# 数据的标准差小于0.1倍的平均值，highlight = std / mean, sig = mean / (mean + std)
def is_Evenness(dataset):
    flag, highlight, sig, values = False, None, None, list(dataset.values())
    if len(values) < 3:
        return flag, highlight, sig, values
    mean = np.mean(values)
    std = np.std(values)
    if std < 0.1 * abs(mean):
        flag = True
        highlight = std / mean
        sig = mean / (mean + std)
        values = [(item, dataset[item]) for item in dataset]
    return flag, highlight, sig, values


def is_Outlier(dataset):
    flag, highlight, sig, values = False, None, None, None
    return flag, highlight, sig, values


# 必须为时间相关的维度，计算方式与Shape相同
def is_Trend(dataset):
    flag, highlight, sig, values = False, None, None, None
    sorted_keys = sorted(dataset.keys())
    if len(sorted_keys) <= 3:
        return flag, highlight, sig, values
    y = [dataset[key] for key in sorted_keys]
    x = [i + 1 for i in range(len(y))]
    linear_coefficient = np.polyfit(x, y, 1)
    linear_fitted_val = np.polyval(linear_coefficient, x)
    r2 = r2_score(y, linear_fitted_val, multioutput='raw_values')
    probability = get_probability_normal_distribution(linear_coefficient[0], mean=0.2, std=2)
    probability = 2 * min(probability, 1 - probability)
    if r2 * (1 - probability) >= 0.5:
        flag, highlight, sig, values = True, linear_coefficient[0], float(r2 * (1 - probability)), [(key, dataset[key]) for key in sorted_keys]
    return flag, highlight, sig, values


def dp(dataset, insight_type, total):
    flag = False
    highlight = None
    imp = 0
    sig = 0
    values = []
    if insight_type == 'Attribution':
        flag, highlight, sig, values = is_Attribution(dataset)
    elif insight_type == 'Outstanding No.1':
        flag, highlight, sig, values = is_Outstanding_No1(dataset)
    elif insight_type == 'Outstanding No.2':
        flag, highlight, sig, values = is_Outstanding_No2(dataset)
    elif insight_type == 'Outstanding Last':
        flag, highlight, sig, values = is_Outstanding_Last(dataset)
    elif insight_type == 'Evenness':
        flag, highlight, sig, values = is_Evenness(dataset)
    elif insight_type == 'Change Point':
        flag, highlight, sig, values = is_Change_Point(dataset)
    elif insight_type == 'Outlier':
        flag, highlight, sig, values = is_Outlier(dataset)
    elif insight_type == 'Trend':
        flag, highlight, sig, values = is_Trend(dataset)

    count = 0
    if flag:
        for item in values:
            count += item[1]
        imp = count / total
    return flag, highlight, imp, sig, values


def subspace_extending(table_name, insight_type, subspace, breakdown, exd_dim, measure, measure_col, result_path):
    print('Subspace extending for', table_name, insight_type)
    start_time = time.perf_counter()
    data_scope, total = construct_HDS(subspace, breakdown, exd_dim, measure, measure_col, table_name)
    commonness = construct_HDP(data_scope, insight_type, total)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    commonness_file = open(result_path + table_name + '_' + insight_type + '.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(commonness_file)
    csv_writer.writerow(['Exd_Type', 'Exd_Dimension', 'Subspace', 'Breakdown', 'Measure', 'Insight_Type', 'Highlight', 'Imp', 'Sig', 'Score', 'Values'])
    satisfying_current_type = False
    subspace_copy = {key: subspace[key] for key in subspace.keys()}
    for item in commonness:
        subspace_copy[exd_dim] = item[0]
        if subspace == subspace_copy:
            satisfying_current_type = True
            csv_writer.writerow(['Seed Insight', exd_dim, subspace, breakdown, measure, str(item[1]), str(item[2]), str(item[3]), str(item[4]), str(item[5]), str(item[6])])
            commonness.remove(item)
    if not satisfying_current_type:
        csv_writer.writerow(
            ['Seed Insight', exd_dim, subspace, breakdown, measure, insight_type, None, None, None, None, None])
    for item in commonness:
        subspace_copy[exd_dim] = item[0]
        csv_writer.writerow(['Subspace Extending', exd_dim, subspace_copy, breakdown, measure, str(item[1]), str(item[2]), str(item[3]), str(item[4]), str(item[5]), str(item[6])])
    commonness_file.close()
    print('Subspace extending down!\nTime:', str(time.perf_counter() - start_time) + 's')


if __name__ == '__main__':
    for T in insight_types:
        subspace_extending(table_name='country',
                           insight_type=T,
                           subspace={'year': '*', 'region': '*', 'country': 'Iraq'},
                           breakdown='year',
                           exd_dim='country',
                           measure='sum(count)',
                           measure_col='count',
                           result_path='../result/commonness/subspace_extending/')
    print('Done!!!')
