import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
from uf import *


class GXDataset_dn(Dataset):
    def __init__(self, path, t='train', category=None):
        """Init function."""
        data_x = []
        data_y = []
        # 类别 {'0V': 1, '0.25V': 2, '0.5V': 3, '0.75V': 4, '1V': 5, '1.25V': 6, '1.5V': 7, '5V': 8}

        index = []

        if t == 'train':
            for i in range(1, 9):
                # 读取 .pkl 文件
                with open(path + '/data_list_x_{}.pkl'.format(i), 'rb') as f:
                    data = pickle.load(f)
                    data_x.extend(data[:400])

                # 读取 .pkl 文件
                with open(path + '/data_list_y_{}.pkl'.format(i), 'rb') as f:
                    data = pickle.load(f)
                    data_y.extend(data[:400])
        elif category is None:
            for i in range(1, 9):
                # 读取 .pkl 文件
                with open(path + '/data_list_x_{}.pkl'.format(i), 'rb') as f:
                    data = pickle.load(f)
                    data_x.extend(data[400:])

                # 读取 .pkl 文件
                with open(path + '/data_list_y_{}.pkl'.format(i), 'rb') as f:
                    data = pickle.load(f)
                    data_y.extend(data[400:])
        else:
            i = category
            # 读取 .pkl 文件
            with open(path + '/data_list_x_{}.pkl'.format(i), 'rb') as f:
                data = pickle.load(f)
                data_x.extend(data[400:])

            # 读取 .pkl 文件
            with open(path + '/data_list_y_{}.pkl'.format(i), 'rb') as f:
                data = pickle.load(f)
                data_y.extend(data[400:])

            # 读取 .pkl 文件  index 是原始文件名
            with open(path + '/index_{}.pkl'.format(i), 'rb') as f:
                data = pickle.load(f)
                index.extend(data[400:])

        self.data_x = data_x
        self.data_y = data_y
        self.index = index

    def __getitem__(self, index):
        """Get item."""
        index_ = index
        # get the filename
        data_x = self.data_x[index_][2000:8000]
        data_y = self.data_y[index_][2000:8000]

        data_x = np.array(data_x, dtype=np.float32) * np.pi / 180  # 转化弧度
        data_x = data_x - np.mean(data_x)  # 均值归0
        data_y = np.array(data_y, dtype=np.float32)
        data_y = wt(data_y)

        if 'impulsive' in self.index[index_]:
            i_or_m = 0
        else:
            i_or_m = 1

        return torch.from_numpy(np.array(data_x, dtype=np.float32)), torch.from_numpy(
            np.array(data_y, dtype=np.float32)), i_or_m

    def __len__(self):
        """Length."""
        return len(self.data_x)


if __name__ == '__main__':

    data_set = GXDataset_dn('./ndata/data_dn', t='test')
    DataLoader = torch.utils.data.DataLoader(data_set, batch_size=8, shuffle=True, drop_last=True)
    print(len(DataLoader))
    for dx, dy, im in DataLoader:
        print('-------------------')
        print(dx.shape)
        print(dx)
        print(dy.shape)
        print(dy)
        plot_dn(dx[0], dx[0], dy[0])
        wt(dy[0])
        print(im)
