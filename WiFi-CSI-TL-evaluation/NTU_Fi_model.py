import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class NTU_Fi_LSTM(nn.Module):
    def __init__(self,num_classes):
        super(NTU_Fi_LSTM,self).__init__()
        self.lstm = nn.LSTM(342,64,num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,342,500)
        x = x.permute(2,0,1)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class NTU_Fi_BiLSTM(nn.Module):
    def __init__(self,num_classes):
        super(NTU_Fi_BiLSTM,self).__init__()
        self.lstm = nn.LSTM(342,64,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,342,500)
        x = x.permute(2,0,1)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class NTU_Fi_CNN_GRU(nn.Module):
    def __init__(self, num_classes, freeze_cnn=False):
        super(NTU_Fi_CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,16,12,6),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,7,3),
            nn.ReLU(),
        )
        self.mean = nn.AvgPool1d(32)
        self.gru = nn.GRU(8,128,num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128,num_classes),
            nn.Softmax(dim=1)
        )
        if freeze_cnn:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self,x):
        batch_size = len(x)
        # batch x 3 x 114 x 500
        x = x.view(batch_size,3*114,500)
        x = x.permute(0,2,1)
        # batch x 500 x 342
        x = x.reshape(batch_size*500,1, 3*114)
        # (batch x 500) x 1 x 342
        x = self.encoder(x)
        # (batch x 500) x 32 x 8
        x = x.permute(0,2,1)
        x = self.mean(x)
        x = x.reshape(batch_size, 500, 8)
        # batch x 500 x 8
        x = x.permute(1,0,2)
        # 500 x batch x 8
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs



class NTU_Fi_CNN_BiLSTM_Migration(nn.Module):
    def __init__(self, num_classes, freeze_cnn=False):
        super(NTU_Fi_CNN_BiLSTM_Migration, self).__init__()

        # CNN模块，用于特征提取
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=12, stride=6),  # 1 input channel, 16 output channels
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(16, 32, kernel_size=7, stride=3),  # 16 input channels, 32 output channels
            nn.ReLU(),
        )

        # 计算CNN输出的维度后进行LSTM的输入，8是CNN特征的尺寸
        self.lstm_input_size = 32  # CNN输出通道数
        self.lstm_hidden_size = 64
        self.num_layers = 1

        # LSTM模块，处理时序特征
        self.lstm = nn.LSTM(32, 64, 1, batch_first=True, bidirectional=True)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(self.lstm_hidden_size * 2, num_classes),
            nn.Softmax(dim=1)  # 输出类别概率
        )

        # 是否冻结CNN层
        if freeze_cnn:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        batch_size = x.size(0)

        # CNN部分 - 特征提取
        x = x.view(batch_size, 1, -1)  # 调整输入形状 (batch_size, 1, feature_size)
        x = self.encoder(x)  # 输出大小：(batch_size, 32, new_length)

        # LSTM部分 - 处理时序数据
        x = x.permute(0, 2, 1)  # (batch_size, new_length, 32)
        lstm_out, (ht, ct) = self.lstm(x)  # ht: 最后一个时刻的hidden state

        final_output = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)
        # 使用LSTM最后一层的hidden state进行分类
        outputs = self.classifier(final_output)  # ht[-1]是最后一层LSTM的输出

        return outputs

class NTU_Fi_CNN_LSTM(nn.Module):
    def __init__(self, num_classes, freeze_cnn=False):
        super(NTU_Fi_CNN_LSTM, self).__init__()

        # CNN模块，用于特征提取
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=12, stride=6),  # 1 input channel, 16 output channels
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 2x MaxPool
            nn.Conv1d(16, 32, kernel_size=7, stride=3),  # 16 input channels, 32 output channels
            nn.ReLU(),
        )

        # 计算CNN输出的维度后进行LSTM的输入，8是CNN特征的尺寸
        self.lstm_input_size = 32  # CNN输出通道数
        self.lstm_hidden_size = 64
        self.num_layers = 1

        # LSTM模块，处理时序特征
        self.lstm = nn.LSTM(32, 64, 1, batch_first=True)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(self.lstm_hidden_size, num_classes),  # LSTM输出的hidden_size作为全连接层输入
            nn.Softmax(dim=1)  # 输出类别概率
        )

        # 是否冻结CNN层
        if freeze_cnn:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        batch_size = x.size(0)

        # CNN部分 - 特征提取
        x = x.view(batch_size, 1, -1)  # 调整输入形状 (batch_size, 1, feature_size)
        x = self.encoder(x)  # 输出大小：(batch_size, 32, new_length)

        # LSTM部分 - 处理时序数据
        x = x.permute(0, 2, 1)  # (batch_size, new_length, 32)
        lstm_out, (ht, ct) = self.lstm(x)  # ht: 最后一个时刻的hidden state

        # 使用LSTM最后一层的hidden state进行分类
        outputs = self.classifier(ht[-1])  # ht[-1]是最后一层LSTM的输出

        return outputs


class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DualConv, self).__init__()
        # 这里只做一个简单示例，你可以实现更复杂的双分支卷积结构
        self.conv_a = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(stride,1), padding=1, bias=False)
        self.conv_b = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out_a = self.conv_a(x)
        out_b = self.conv_b(x)
        out = out_a + out_b
        out = self.bn(out)
        out = self.relu(out)
        return out

class Block_dual(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block_dual, self).__init__()

        # 使用 DualConv 模块替换原有的卷积层
        self.conv1 = DualConv(in_channels, out_channels, stride=stride)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        # 使用 DualConv 模块替换原有的卷积层
        self.conv2 = DualConv(out_channels, out_channels, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class NTU_Fi_ResNet_GRU(nn.Module):
    def __init__(self, num_classes, freeze_cnn=False):
        super(NTU_Fi_ResNet_GRU, self).__init__()
        # 调整 reshape 部分，保留时间信息
        self.reshape = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(15, 1), stride=(3, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=1),
            nn.ReLU()
        )
        self.in_channels = 32

        # 第一层卷积
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        # 使用 Block_dual 构建 ResNet 的四个残差层
        self.layer1 = self._make_layer(Block_dual, 32, layers=2, stride=1)
        self.layer2 = self._make_layer(Block_dual, 64, layers=2, stride=2)
        self.layer3 = self._make_layer(Block_dual, 128, layers=2, stride=2)
        self.layer4 = self._make_layer(Block_dual, 256, layers=2, stride=2)

        # GRU 部分，假设经过 ResNet 后特征图高度压缩为 1
        # 经过卷积操作后，特征通道为 256，时间维度保持不变
        self.gru = nn.GRU(256, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

        if freeze_cnn:
            for param in self.reshape.parameters():
                param.requires_grad = False
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in layer.parameters():
                    param.requires_grad = False

    def _make_layer(self, block, planes, layers, stride):
        downsample = None
        if stride != 1 or self.in_channels != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(planes)
            )
        layers_list = []
        layers_list.append(block(self.in_channels, planes, i_downsample=downsample, stride=stride))
        self.in_channels = planes
        for _ in range(1, layers):
            layers_list.append(block(planes, planes))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        # x: (batch, 3, height, time)
        x = self.reshape(x)          # (batch, 32, new_height, time)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (batch, 256, H, time)
        # 假设 H 压缩为1，则 squeeze 掉高度维度
        x = x.squeeze(2)    # (batch, 256, time)
        x = x.permute(0, 2, 1)  # (batch, time, 256)
        gru_out, _ = self.gru(x)  # gru_out: (batch, time, 128)
        out = gru_out[:, -1, :]   # 取最后一个时间步
        out = self.classifier(out)
        return out
