import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTNetForClassification(nn.Module):
    def __init__(self, args, data):
        super(LSTNetForClassification, self).__init()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = (self.P - self.Ck) // self.skip
        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)

        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, args.num_classes)  # Output num_classes for classification
        else:
            self.linear1 = nn.Linear(self.hidR, args.num_classes)  # Output num_classes for classification

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, args.num_classes)  # Output num_classes for classification

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
    def __init__(self, input_channels, output_size, dropout_rate):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 100, kernel_size=50)
        self.conv2 = nn.Conv1d(100, 100, kernel_size=50)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100 * (151 - 50 * 2 + 2), 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
    # Permute dimensions to [batch_size, channels, sequence_length]
        x = x.permute(0, 2, 1)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        self.fc1 = nn.Linear(x.view(x.size(0), -1).shape[1], 100)
    
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


        # RNN
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out
    


