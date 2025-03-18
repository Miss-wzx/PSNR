import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 编码器模块
class Encoder(nn.Module):
    def __init__(self, input_dim=10000, conv_channels=None):
        super(Encoder, self).__init__()
        if conv_channels is None:
            conv_channels = [16, 32, 64]
        self.conv1 = nn.Conv1d(1, conv_channels[0], kernel_size=10, stride=5, padding=3)  # 一维卷积
        self.bn1 = nn.BatchNorm1d(conv_channels[0])  # 批归一化
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(conv_channels[2])
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局池化

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度 (B, 1, 10000)
        x = F.relu(self.bn1(self.conv1(x)))  # 卷积 + 批归一化 + 激活
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)  # (B, 64, 1)
        x = x.squeeze(-1)  # 去掉最后的 1 维，变成 (B, 64)
        return x


# 分类器模块
class Classifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=8):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # 全连接层
        self.dropout = nn.Dropout(0.3)  # 随机失活率 30%
        self.fc2 = nn.Linear(512, output_dim)  # 全连接层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 激活函数
        x = self.dropout(x)
        x = self.fc2(x)  # 输出层
        c = torch.sigmoid(x)
        return c


# EC完整模型
class EC(nn.Module):
    def __init__(self, input_dim=10000, conv_channels=None, output_dim=8):
        super(EC, self).__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 256]
        self.encoder = Encoder(input_dim=input_dim, conv_channels=conv_channels)
        self.classifier = Classifier(input_dim=conv_channels[-1], output_dim=output_dim)

    def forward(self, x):
        features = self.encoder(x)  # 编码器提取特征
        c = self.classifier(features)  # 分类器分类
        return c, features


# 生成器
class LGenerator(nn.Module):
    def __init__(self, input_dim=10000, conv_channels=None, output_dim=10000):
        super(LGenerator, self).__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 256]
        self.encoder = Encoder(input_dim=input_dim, conv_channels=conv_channels)

        self.fc1 = nn.Linear(conv_channels[-1], 512)  # 全连接层将特征扩展
        self.fc2 = nn.Linear(512, 1024)  # 增加非线性表达能力
        self.fc3 = nn.Linear(1024, output_dim)  # 输出去噪信号

    def forward(self, x):
        features = self.encoder(x)  # 编码器提取特征

        x = F.relu(self.fc1(features))  # 激活函数
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 输出范围限制为 [-1, 1]，匹配信号范围
        return x


# CONV生成器
class ConvGenerator(nn.Module):
    def __init__(self, input_dim=10000, conv_channels=None, output_dim=10000):
        super(ConvGenerator, self).__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 256]
        self.encoder = Encoder(input_dim=input_dim, conv_channels=conv_channels)

        self.conv1 = nn.ConvTranspose1d(conv_channels[-1], 128, kernel_size=8, stride=4, padding=1)  # 反卷积
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.ConvTranspose1d(128, 64, kernel_size=16, stride=8, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.ConvTranspose1d(64, 32, kernel_size=16, stride=8, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.ConvTranspose1d(32, 1, kernel_size=32, stride=16, padding=2)  # 输出到信号长度

        self.fc1 = nn.Linear(7020, 8192)  # 全连接层，将输入特征扩展到初始大小
        self.fc2 = nn.Linear(8192, output_dim)  # 输出去噪信号

        self.tanh = nn.Tanh()  # 将输出限制在 [-1, 1] 范围内

    def forward(self, x):
        features = self.encoder(x)  # 编码器提取特征
        x = features.view(features.size(0), 256, 1)  # 调整为 (B, 256, 1)
        x = F.relu(self.bn1(self.conv1(x)))  # 反卷积 + BN + 激活
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.squeeze(1)  # 移除通道维度，变成 (B, 7020)
        x = F.relu(self.fc1(x))  # 激活函数
        x = self.fc2(x)
        x = self.tanh(x) * np.pi / 2
        return x


# CONV生成器
class ConvGenerator_dn(nn.Module):
    def __init__(self, input_dim=6000, conv_channels=None, output_dim=6000):
        super(ConvGenerator_dn, self).__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 256]
        self.encoder = Encoder(input_dim=input_dim, conv_channels=conv_channels)

        self.conv1 = nn.ConvTranspose1d(conv_channels[-1], 128, kernel_size=8, stride=4, padding=1)  # 反卷积
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.ConvTranspose1d(128, 64, kernel_size=16, stride=8, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.ConvTranspose1d(64, 32, kernel_size=16, stride=8, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.ConvTranspose1d(32, 1, kernel_size=32, stride=16, padding=2)  # 输出到信号长度

        self.fc1 = nn.Linear(7020, 8192)  # 全连接层，将输入特征扩展到初始大小
        self.fc2 = nn.Linear(8192, output_dim)  # 输出去噪信号

        self.tanh = nn.Tanh()  # 将输出限制在 [-1, 1] 范围内

    def forward(self, x):
        features = self.encoder(x)  # 编码器提取特征
        x = features.view(features.size(0), 256, 1)  # 调整为 (B, 256, 1)
        x = F.relu(self.bn1(self.conv1(x)))  # 反卷积 + BN + 激活
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.squeeze(1)  # 移除通道维度，变成 (B, 7020)
        x = F.relu(self.fc1(x))  # 激活函数
        x = self.fc2(x)
        x = self.tanh(x) * np.pi / 2
        return x

# 辨别器
class Discriminator(nn.Module):
    def __init__(self, input_dim=10000):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=3)  # 一维卷积
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 512, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(512)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 1024)  # 二分类输出
        self.fc2 = nn.Linear(1024, 256)  # 二分类输出
        self.fc3 = nn.Linear(256, 1)  # 二分类输出

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # 去掉最后一维，变成 (B, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)  # 输出概率值
        return x


if __name__ == '__main__':
    # 测试模型
    model = EC()
    dummy_input = torch.randn(8, 10000)  # Batch size = 8
    output, f = model(dummy_input)
    print("Output shape:", f.shape)

    # netG = LGenerator()
    netG = ConvGenerator()
    netD = Discriminator()

    g = netG(dummy_input)
    print(g.shape)

    d = netD(g)
    print(d)
