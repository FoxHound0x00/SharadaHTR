import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.resblocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2)
        )
        self.conv2 = nn.Conv2d(256, hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resblocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        T, b, h = out.size() # T - time_steps
        # print(out.size())
        # out = self.fc(out[:, -1, :])

        t_rec = out.reshape(T * b, h)
        out = self.fc(t_rec) # [T * b, nOut]
        out = out.reshape(T, b, -1)

        return out

class CRNN(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, num_classes):
        super(CRNN, self).__init__()
        self.cnn = CNN(input_channels, hidden_size)
        self.rnn = BiLSTM(hidden_size, hidden_size, num_layers, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        # print("shape",x.shape)
        x = x.squeeze(2)
        # print("after squeeze", x.shape )
        x = x.permute(2, 0, 1)
        x = self.rnn(x)
        x = torch.nn.functional.log_softmax(x, dim=2) # Behaves Differently when batch_first is specified https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # print(x.shape)
        return x