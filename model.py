import torch.nn as nn


class CaptchaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3,), padding=(1, 1))
        self.mp1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.mp2 = nn.MaxPool2d((2, 2))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.l1 = nn.Linear(1152, 64)
        self.do1 = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(64, 32, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
        self.gru1 = nn.GRU(64, 32, num_layers=3, batch_first=True, dropout=0.2, bidirectional=True)
        self.l2 = nn.Linear(64, 20)

    def forward(self, input):
        x = self.relu1(self.conv1(input))
        x = self.mp1(x)
        x = self.relu2(self.conv2(x))
        x = self.mp2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.do1(self.relu3(self.l1(x)))
        x, _ = self.gru1(x)
        x = self.l2(x)
        return x
