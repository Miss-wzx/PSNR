import os
import numpy as np
from history import History
import pickle
import matplotlib.pyplot as plt
from dataset_dn import GXDataset_dn
import torch
from torch.utils.data import DataLoader
from uf import *
from module import *


# 平滑Loss曲线
def average_every_n_elements(data, n):
    return [sum(data[i: i + n]) / n for i in range(0, len(data), n)]


# 查看训练历史
def pl(f):
    h = pickle.loads(open(f, 'rb').read())
    plt.plot(h.train_loss, label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.legend()
    plt.show()


# 应用网络
def app(pth, c=None):
    print('app_net')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.is_available())

    # 读取数据
    data_path = './ndata/data_dn'

    test_dataset = GXDataset_dn(data_path, t='test', category=c)
    test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=False)
    # 实例化模型
    m = ConvGenerator_dn().to(device)

    # 加载模型参数
    if os.path.exists(pth):
        print('加载已训练模型参数' + '-' * 30)
        m.load_state_dict(torch.load(pth, weights_only=True))  # 仅加载参数

    m = m.eval()
    nsl = []  # 原始信号
    tsl = []  # 小波信号 监督
    gsl = []  # 降噪信号
    iml = []  # 激励电压类型

    for batch_id, (ns, ts, im) in enumerate(test_data_loader, 0):
        ns, ts = ns.to(device), ts.to(device)
        gs = m(ns)
        nsl.extend(ns.tolist())
        tsl.extend(ts.tolist())
        iml.extend(im.tolist())
        gsl.extend(gs.tolist())

    return nsl, gsl, tsl, iml


if __name__ == '__main__':
    # from torchsummary import summary
    # 查看模型摘要
    # m = ConvGenerator_dn()
    # summary(m, (6000,))

    # pl('./save/CGdn.pkl')  # 查看Loss曲线

    ld = {1: '0V', 2: '0.25V', 3: '0.5V', 4: '0.75V', 5: '1V', 6: '1.25V', 7: '1.5V', 8: '5V'}  # 标签

    start_index = 1300
    end_index = 3300

    ns_ses = []
    gs_ses = []
    ts_ses = []

    # 误差图
    datan = {'category': [], 'value': []}
    datag = {'category': [], 'value': []}
    datat = {'category': [], 'value': []}

    # 绘制图像 2000 长度
    for idx in [1, 2, 3, 4, 5, 6]:  # 0V-1.25V
        print(idx)
        ns_list, gs_list, ts_list, im_list = app('./model/CGdn.pth', c=idx)
        for i in range(10):  # 10张

            # 绘制样本图6x1
            plot_sample(ns_list[i][start_index:end_index], fs='./img/样本图/dn_{}_{}.png'.format(ld[idx], i),
                        show=False)

            # 绘制降噪效果图6x3
            plot_dn_nre(ns_list[i][start_index:end_index], gs_list[i][start_index:end_index],
                        ts_list[i][start_index:end_index], fs='./img/降噪效果图/dn_{}_{}.png'.format(ld[idx], i),
                        show=False)

            # 频谱图6x3
            plot_dn_spectrum(ns_list[i], gs_list[i], ts_list[i], fs_file='./img/频谱图/dn_{}_{}.png'.format(ld[idx], i),
                             show=False)

        # 计算频谱平滑度
        ns_ses.append(spectral_smoothness(ns_list))
        ts_ses.append(spectral_smoothness(ts_list))
        gs_ses.append(spectral_smoothness(gs_list))

        # 计算误差图
        category, value = calculation_error_chart(ns_list, im_list, idx)
        datan['category'].extend(category)
        datan['value'].extend(value)

        category, value = calculation_error_chart(ts_list, im_list, idx)
        datat['category'].extend(category)
        datat['value'].extend(value)

        category, value = calculation_error_chart(gs_list, im_list, idx)
        datag['category'].extend(category)
        datag['value'].extend(value)

    print('---------------------------------------------------------')
    print('---------------------------------------------------------')

    print('原始信号', ns_ses)
    print('小波信号', ts_ses)
    print('降噪信号', gs_ses)
    print(np.mean(np.array(ns_ses)))
    print(np.mean(np.array(ts_ses)))
    print(np.mean(np.array(gs_ses)))

    print('---------------------------------------------------------')
    print('---------------------------------------------------------')

    # 计算误差图
    # for idx in [1, 2, 3, 4, 5, 6]:  # 0V-1.25V [0, 1.75, 3.5, 5.25, 7, 8.75] nm 形变
    #     ns_list, gs_list, ts_list, im_list = app('./model/CGdn.pth', c=idx)
    #
    #     category, value = calculation_error_chart(ns_list, im_list, idx)
    #     datan['category'].extend(category)
    #     datan['value'].extend(value)
    #
    #     category, value = calculation_error_chart(ts_list, im_list, idx)
    #     datat['category'].extend(category)
    #     datat['value'].extend(value)
    #
    #     category, value = calculation_error_chart(gs_list, im_list, idx)
    #     datag['category'].extend(category)
    #     datag['value'].extend(value)

    # 形变计算均值 y_pred
    y_pred = plot_calculation_error_chart(datan, './img/形变复原图/n.png')
    print(y_pred)
    # 计算均方误差（MSE）
    mse = np.mean((np.array([0, 1.75, 3.5, 5.25, 7, 8.75]) - np.array(y_pred)) ** 2)
    print(f"均方误差 (MSE) n: {mse, np.sqrt(mse)}")

    y_pred = plot_calculation_error_chart(datat, './img/形变复原图/t.png')
    print(y_pred)
    # 计算均方误差（MSE）
    mse = np.mean((np.array([0, 1.75, 3.5, 5.25, 7, 8.75]) - np.array(y_pred)) ** 2)
    print(f"均方误差 (MSE) t: {mse, np.sqrt(mse)}")

    y_pred = plot_calculation_error_chart(datag, './img/形变复原图/g.png')
    print(y_pred)
    # 计算均方误差（MSE）
    mse = np.mean((np.array([0, 1.75, 3.5, 5.25, 7, 8.75]) - np.array(y_pred)) ** 2)
    print(f"均方误差 (MSE) g: {mse, np.sqrt(mse)}")
