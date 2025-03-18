import os
import random
import pickle
import re
from collections import Counter

from uf import *

label_dict = {'0V': 1, '0.25V': 2, '0.5V': 3, '0.75V': 4, '1V': 5, '1.25V': 6, '1.5V': 7, '5V': 8}


def get_filtered_files(fp):
    # 获取文件夹中的所有文件路径
    files = []
    # 定义正则表达式，匹配 I1.txt, I2.txt, I3.txt, I4.txt, I5.txt 等文件
    pattern = r'.*I[1-5]\.txt$'

    # 遍历文件夹中的文件
    for file in os.listdir(fp):
        if re.match(pattern, file):  # 如果文件名符合正则表达式
            files.append(os.path.join(fp, file))

    # 随机打乱文件路径列表
    random.shuffle(files)
    random.shuffle(files)

    return files


def get_data(fp_list):
    data_list_x = []
    data_list_y = []

    for file_path in fp_list:
        try:

            I = read_numbers_from_file(file_path)

            Q = read_numbers_from_file(file_path.replace('I', 'Q'))

            data = calculate_t(Q, I)

            if 'impulsive' in file_path:
                data_list_x.append([-d for d in data])  # 将文件内容保存到列表
            else:
                data_list_x.append(data)  # 将文件内容保存到列表

            S = read_numbers_from_file(file_path.replace('I', 'sit'))

            data_list_y.append(S)  # 将文件内容保存到列表

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return data_list_x, data_list_y


# 遍历文件夹中的所有项
for item in os.listdir('./'):
    item_path = os.path.join('./', item)

    # 如果是文件夹，添加到列表中
    if os.path.isdir(item_path) and item[0].isdigit():
        print(item_path)
        # 获取筛选后的文件路径
        file_paths = get_filtered_files(item_path)

        # 统计每个元素的出现次数
        counter = Counter(file_paths)

        # 输出重复的元素
        duplicates = {key: count for key, count in counter.items() if count > 1}
        if duplicates:
            print("Duplicates found:", duplicates)
        else:
            print("No duplicates found.")
            if len(file_paths) == 500:
                # 保存为 pkl 文件
                with open('./data_dn/index_{}.pkl'.format(label_dict[item]), 'wb') as f:
                    pickle.dump(file_paths, f)

                data_list_x, data_list_y = get_data(file_paths)

                with open('./data_dn/data_list_x_{}.pkl'.format(label_dict[item]), 'wb') as f:
                    pickle.dump(data_list_x, f)

                with open('./data_dn/data_list_y_{}.pkl'.format(label_dict[item]), 'wb') as f:
                    pickle.dump(data_list_y, f)

                # 输出打乱后的文件路径列表
                print(file_paths)
