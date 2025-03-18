import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt

plt.rcParams['mathtext.fontset'] = 'custom'  # 允许自定义数学字体
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 设置数学斜体字体
plt.rcParams['font.family'] = 'Times New Roman'  # 设定整体字体


# 绘制样本图
def plot_sample(T, fs='plot.png', show=True):
    y_min = -0.2
    y_max = 0.2
    lw = 0.3

    fig = plt.figure(figsize=(8, 2.4))  # 创建画布

    # 绘制原始信号
    plt.plot(T, label='Before noise reduction', color='black', linewidth=lw)
    # $\mathregular{\theta(t)}$
    plt.xlabel(r'Original Signal $\theta(t)$ (ms)', fontdict={'size': 22}, fontfamily="Times New Roman")
    plt.ylabel('Phase (rad)', fontdict={'size': 22}, fontfamily="Times New Roman", labelpad=5)
    # plt.grid()
    plt.xlim(0, len(T))  # 设置 X 轴范围
    plt.ylim(y_min - 0.03, y_max + 0.03)  # 设置 Y 轴范围

    # 设置 X 和 Y 轴刻度字体、间距
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks([0, 500, 1000, 1500, 2000], [0, 0.1, 0.2, 0.3, 0.4], fontfamily="Times New Roman")
    plt.yticks([y_min, 0, y_max], [y_min, 0, y_max], fontfamily="Times New Roman")

    plt.tight_layout()  # 自动调整子图间距
    if show:
        plt.show()
    else:
        fig.savefig(fs, dpi=800, bbox_inches='tight')
    plt.close(fig)  # 关闭当前 figure
    # plt.close('all')  # 关闭所有已打开的 figure


# 绘制降噪效果图
def plot_dn_nre(T, Tdn, Tdnt, fs='plot.png', show=True):
    y_min = -0.2
    y_max = 0.2

    lw = 0.3

    fig, axs = plt.subplots(3, 1, figsize=(8, 7.2))  # 创建 3 行 1 列的子图

    # 绘制 原始信号
    axs[0].plot(T, label='Before noise reduction', color='black', linewidth=lw)
    axs[0].set_xlabel(r'Original Signal $\theta(t)$ (ms)', fontdict={'size': 22}, fontfamily="Times New Roman")
    axs[0].set_ylabel('Phase (rad)', fontdict={'size': 22}, fontfamily="Times New Roman", labelpad=5)
    # axs[0].grid()
    axs[0].set_xlim(0, len(T))  # 设置 X 轴范围
    axs[0].set_ylim(y_min - 0.03, y_max + 0.03)  # 设置 Y 轴范围
    # 设置 X 和 Y 轴刻度字体、间距
    axs[0].tick_params(axis='x', labelsize=18)
    axs[0].tick_params(axis='y', labelsize=18)
    axs[0].set_xticks([0, 500, 1000, 1500, 2000])  # 设置 Y 轴刻度位置
    axs[0].set_xticklabels([0, 0.1, 0.2, 0.3, 0.4], fontfamily="Times New Roman")
    # 手动设置 Y 轴的刻度
    axs[0].set_yticks([y_min, 0, y_max])  # 设置 Y 轴刻度位置
    # 格式化 Y 轴标签
    axs[0].set_yticklabels([y_min, 0, y_max], fontfamily="Times New Roman")

    # 绘制 降噪信号
    axs[1].plot(Tdnt, label='After noise reduction', color='black', linewidth=lw + 0.3)
    axs[1].set_xlabel(r'Wavelet transform $\theta(t)$ (ms)', fontdict={'size': 22}, fontfamily="Times New Roman")
    axs[1].set_ylabel('Phase (rad)', fontdict={'size': 22}, fontfamily="Times New Roman", labelpad=5)
    # axs[1].grid()
    axs[1].set_xlim(0, len(Tdnt))  # 设置 X 轴范围
    axs[1].set_ylim(y_min - 0.03, y_max + 0.03)  # 设置 Y 轴范围
    # 设置 X 和 Y 轴刻度字体、间距
    axs[1].tick_params(axis='x', labelsize=18)
    axs[1].tick_params(axis='y', labelsize=18)
    axs[1].set_xticks([0, 500, 1000, 1500, 2000])  # 设置 Y 轴刻度位置
    axs[1].set_xticklabels([0, 0.1, 0.2, 0.3, 0.4], fontfamily="Times New Roman")
    # 手动设置 Y 轴的刻度
    axs[1].set_yticks([y_min, 0, y_max])  # 设置 Y 轴刻度位置
    # 格式化 Y 轴标签
    axs[1].set_yticklabels([y_min, 0, y_max], fontfamily="Times New Roman")

    # 绘制 真值信号
    axs[2].plot(Tdn, label='Real', color='black', linewidth=lw + 0.3)
    axs[2].set_xlabel(r'CNN $\theta(t)$ (ms)', fontdict={'size': 22}, fontfamily="Times New Roman")
    axs[2].set_ylabel('Phase (rad)', fontdict={'size': 22}, fontfamily="Times New Roman", labelpad=5)
    # axs[2].grid()
    axs[2].set_xlim(0, len(Tdn))  # 设置 X 轴范围
    axs[2].set_ylim(y_min - 0.03, y_max + 0.03)  # 设置 Y 轴范围
    # 设置 X 和 Y 轴刻度字体、间距
    axs[2].tick_params(axis='x', labelsize=18)
    axs[2].tick_params(axis='y', labelsize=18)
    axs[2].set_xticks([0, 500, 1000, 1500, 2000])  # 设置 Y 轴刻度位置
    axs[2].set_xticklabels([0, 0.1, 0.2, 0.3, 0.4], fontfamily="Times New Roman")
    # 手动设置 Y 轴的刻度
    axs[2].set_yticks([y_min, 0, y_max])  # 设置 Y 轴刻度位置
    # 格式化 Y 轴标签
    axs[2].set_yticklabels([y_min, 0, y_max], fontfamily="Times New Roman")

    plt.tight_layout()  # 自动调整子图间距
    if show:
        plt.show()
    else:
        fig.savefig(fs, dpi=800, bbox_inches='tight')
    plt.close(fig)  # 关闭当前 figure


# 绘制频谱图 fs 采样频率 5MHz
def plot_dn_spectrum(T, Tdn, Tdnt, fs=5000000, fs_file='plot.png', show=True):
    lw = 1
    # y_list = [-40, -15, 10, 35, 60]
    y_list = [-40, 10, 60]

    fig, axs = plt.subplots(3, 1, figsize=(8, 7.2))  # 创建 3 行 1 列的子图

    # 计算频谱函数 正
    def compute_spectrum_z(signal, fs):
        n = len(signal)
        freqs = np.fft.fftfreq(n, 1 / fs)  # 计算频率
        fft_values = np.fft.fft(signal)  # 计算傅里叶变换
        spectrum = np.abs(fft_values)[:n // 2]  # 只取正频率部分
        freqs = freqs[:n // 2]  # 只取正频率部分
        return freqs, spectrum

    # 计算频谱函数
    def compute_spectrum(signal, fs):
        n = len(signal)
        freqs = np.fft.fftfreq(n, 1 / fs)  # 计算频率
        fft_values = np.fft.fft(signal)  # 计算傅里叶变换
        spectrum = np.abs(fft_values)  # 保留负频率部分

        return freqs, 20 * np.log10(spectrum)

    # 绘制原始信号的频谱
    freqs_T, spectrum_T = compute_spectrum(T, fs)
    # print(len(freqs_T))
    axs[0].plot(freqs_T[0:240], spectrum_T[0:240], label='Before noise reduction', color='black', linewidth=lw)
    axs[0].plot(freqs_T[-240:], spectrum_T[-240:], label='Before noise reduction', color='black', linewidth=lw)

    axs[0].set_xlabel('Original signal (kHz)', fontdict={'size': 22}, fontfamily="Times New Roman")
    axs[0].set_ylabel('Magnitude (dB)', fontdict={'size': 22}, fontfamily="Times New Roman", labelpad=5)
    axs[0].set_xlim(-100000, 100000)  # 设置 X 轴范围，负频率部分也需要显示
    axs[0].set_ylim(-45, 65)  # 设置 Y 轴范围
    axs[0].tick_params(axis='x', labelsize=18)
    axs[0].tick_params(axis='y', labelsize=18)
    axs[0].set_xticks([-100000, -50000, 0, 50000, 100000])  # 设置 X 轴刻度位置
    axs[0].set_xticklabels([-100, -50, 0, 50, 100], fontfamily="Times New Roman")
    axs[0].set_yticks(y_list)
    axs[0].set_yticklabels(y_list, fontfamily="Times New Roman")
    axs[0].grid()

    # 绘制降噪信号的频谱
    freqs_Tdnt, spectrum_Tdnt = compute_spectrum(Tdnt, fs)
    axs[1].plot(freqs_Tdnt[0:240], spectrum_Tdnt[0:240], label='After noise reduction', color='black', linewidth=lw)
    axs[1].plot(freqs_Tdnt[-240:], spectrum_Tdnt[-240:], label='After noise reduction', color='black', linewidth=lw)

    axs[1].set_xlabel('Wavelet transform (kHz)', fontdict={'size': 22}, fontfamily="Times New Roman")
    axs[1].set_ylabel('Magnitude (dB)', fontdict={'size': 22}, fontfamily="Times New Roman", labelpad=5)
    axs[1].set_xlim(-100000, 100000)  # 设置 X 轴范围，负频率部分也需要显示
    axs[1].set_ylim(-45, 65)  # 设置 Y 轴范围
    axs[1].tick_params(axis='x', labelsize=18)
    axs[1].tick_params(axis='y', labelsize=18)
    axs[1].set_xticks([-100000, -50000, 0, 50000, 100000])  # 设置 X 轴刻度位置
    axs[1].set_xticklabels([-100, -50, 0, 50, 100], fontfamily="Times New Roman")
    axs[1].set_yticks(y_list)
    axs[1].set_yticklabels(y_list, fontfamily="Times New Roman")
    axs[1].grid()

    # 绘制真值信号的频谱
    freqs_Tdn, spectrum_Tdn = compute_spectrum(Tdn, fs)
    axs[2].plot(freqs_Tdn[0:240], spectrum_Tdn[0:240], label='Real', color='black', linewidth=lw)
    axs[2].plot(freqs_Tdn[-240:], spectrum_Tdn[-240:], label='Real', color='black', linewidth=lw)

    axs[2].set_xlabel('CNN (kHz)', fontdict={'size': 22}, fontfamily="Times New Roman")
    axs[2].set_ylabel('Magnitude (dB)', fontdict={'size': 22}, fontfamily="Times New Roman", labelpad=5)
    axs[2].set_xlim(-100000, 100000)  # 设置 X 轴范围，负频率部分也需要显示
    axs[2].set_ylim(-45, 65)  # 设置 Y 轴范围
    axs[2].tick_params(axis='x', labelsize=18)
    axs[2].tick_params(axis='y', labelsize=18)
    axs[2].set_xticks([-100000, -50000, 0, 50000, 100000])  # 设置 X 轴刻度位置
    axs[2].set_xticklabels([-100, -50, 0, 50, 100], fontfamily="Times New Roman")
    axs[2].set_yticks(y_list)
    axs[2].set_yticklabels(y_list, fontfamily="Times New Roman")
    axs[2].grid()

    plt.tight_layout()  # 自动调整子图间距
    if show:
        plt.show()
    else:
        fig.savefig(fs_file, dpi=800, bbox_inches='tight')
    plt.close(fig)  # 关闭当前 figure


# 小波变换
def wt(noisy_signal):
    # 选择小波基
    # wavelet = 'haar'
    # wavelet = 'db1'
    # wavelet = 'coif1'
    wavelet = 'db4'

    # 进行小波分解
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=2)  # 分解信号 level=2 是分解的层数

    # 阈值处理：使用软阈值方法进行降噪
    threshold = 0.05  # 阈值可以根据需求调整
    coeffs_denoised = []

    # 对每个细节系数进行阈值处理
    for i in range(len(coeffs)):
        if i == 0:
            coeffs_denoised.append(coeffs[i])  # 逼近系数不变
        else:
            coeffs_denoised.append(pywt.threshold(coeffs[i], threshold, mode='soft'))  # 对细节系数应用软阈值

    # 重建降噪后的信号
    denoised_signal = pywt.waverec(coeffs_denoised, wavelet)

    return denoised_signal


# 功率谱
def spectral_entropy(x):
    x = np.asarray(x).flatten()  # 确保是 NumPy 数组并展平
    f, Pxx = signal.welch(x, fs=5e6, nperseg=1024)  # 计算功率谱密度  采样频率5MHz
    Pxx_norm = Pxx / np.sum(Pxx)  # 归一化
    spec_entropy = -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-10))  # 避免 log(0)
    return spec_entropy


# 求频谱平滑度
def spectral_smoothness(ds):
    ses = []
    for d in ds:
        ses.append(spectral_entropy(d))
    return np.mean(np.array(ses))


# 计算误差图
def calculation_error_chart(ds, ims, c):
    ld = {1: '0V', 2: '0.25V', 3: '0.5V', 4: '0.75V', 5: '1V', 6: '1.25V', 7: '1.5V', 8: '5V'}
    category = []
    value = []

    for i, d in enumerate(ds, 0):

        if ims[i] == 0:  # 脉冲
            v = (max(d) - 0) * 95
            category.append(ld[c])
            value.append(v)

        # elif ims[i] == 1:  # 方波
        #     v = (max(d) - 0) * 65
        #     category.append(ld[c])
        #     value.append(v)

        else:
            continue

    return category, value


# 绘制误差图
def plot_calculation_error_chart(data, fs):
    # 设置字体为Times New Roman，字号为12
    plt.rcParams.update({'font.family': 'Times New Roman'})

    df = pd.DataFrame(data)

    # 定义自定义的类别顺序
    category_order = ['0V', '0.25V', '0.5V', '0.75V', '1V', '1.25V']  # 按照指定顺序排列

    # 将 'category' 列转换为一个分类类型，并指定类别顺序
    df['category'] = pd.Categorical(df['category'], categories=category_order, ordered=True)

    # 计算每个类别的均值、最小值和最大值
    mean_values = df.groupby('category')['value'].mean()
    min_values = df.groupby('category')['value'].min()
    max_values = df.groupby('category')['value'].max()

    # 绘制错误条图
    plt.figure(figsize=(9, 5.5))

    # 使用plt.errorbar绘制每个类别的均值和从最小值到最大值的误差条
    categories = mean_values.index
    x_pos = np.arange(len(categories))  # 每个类别的x坐标

    # 绘制散点图（类别与对应的均值）
    plt.scatter(x_pos, [0, 1.75, 3.5, 5.25, 7, 8.75], color='black', s=80, marker='o', label='Real Displacement Value',
                zorder=5)

    # 绘制每个类别的均值和误差条（从最小值到最大值）
    plt.errorbar(x_pos, mean_values, yerr=[mean_values - min_values, max_values - mean_values],
                 fmt='^', color='red', ecolor='red', elinewidth=1.5, capsize=4, markersize=8, capthick=1.5,
                 label='Measurement Value', zorder=6)

    # 设置坐标轴范围
    plt.xlim(-0.5, len(categories) - 0.5)  # x 轴范围设置

    plt.xticks(x_pos, categories, fontsize=24)  # 设置x轴标签为类别

    # 设置y轴刻度位置和标签
    plt.ylim([-5, 35])
    y_ticks_positions = [0, 10, 20, 30]  # 设定刻度位置
    y_ticks_labels = [0, 10, 20, 30]  # 设置对应的标签
    plt.yticks(y_ticks_positions, y_ticks_labels, fontsize=24)  # 设置y轴刻度的位置和标签

    plt.xlabel('Drive Voltage', fontsize=30, labelpad=12)

    plt.ylabel('Displacement (nm)', fontsize=30, labelpad=10)

    plt.tight_layout()  # 自动调整子图间距

    # 绘制背景网格，设置 zorder 使网格显示在散点图下方
    plt.grid(True, zorder=0)

    plt.legend(fontsize=24, loc='upper left')

    plt.savefig(fs, dpi=800, bbox_inches='tight')

    return list(mean_values)
