import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import matplotlib.ticker as m_ticker


# 读取文本文件并将每行转化为数字
def read_numbers_from_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉行末的换行符并转换为浮点数
            try:
                number = float(line.strip())
                numbers.append(number)
            except ValueError:
                print(f"无法转换为数字的行: {line.strip()}")
    return numbers


def gsn(data):
    # 添加高斯噪声
    mean = 0  # 噪声均值
    std_dev = 20  # 噪声标准差
    noise = np.random.normal(mean, std_dev, len(data))  # 生成与数据长度相同的噪声
    # 将噪声添加到数据
    noisy_data = data + noise
    return noisy_data


def calculate_t(Q, I):
    """
    计算 T = arctan(Q/I) 并返回 T 列表。

    参数:
    Q : list
        Q 值列表
    I : list
        I 值列表

    返回:
    T_degrees : list
        T 值（以度为单位）的列表
    """
    # 计算 T
    T = np.arctan2(np.array(Q), np.array(I))
    # 将 T 转换为度
    T_degrees = np.degrees(T)
    # T_degrees = gsn(T_degrees)  # 添加高斯噪声
    return T_degrees.tolist()
