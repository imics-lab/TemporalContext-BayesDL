import torch
import numpy as np
from torch.autograd import Variable

class Data_utility(object):
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2, num_classes=2):
        self.cuda = cuda
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self.num_classes = num_classes
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        self.rse = self._compute_rse(tmp)
        self.rae = self._compute_rae(tmp)

    def _normalized(self, normalize):
        if normalize == 0:
            self.dat = self.rawdat

        if normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)

        if normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.num_classes))  # Modified for classification

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            # For classification, you may need to map the labels to one-hot vectors or other suitable representations
            # Below is a simple example of one-hot encoding for binary classification
            label = int(self.rawdat[idx_set[i], -1])  # Assuming the class label is in the last column
            Y[i, label] = 1

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if self.cuda:
                X = X.cuda()
                Y = Y.cuda()
            yield Variable(X), Variable(Y)

    def _compute_rse(self, data):
        # Compute RSE for classification, e.g., cross-entropy loss
        # Implement the appropriate loss function here
        return 0  # Replace with your implementation

    def _compute_rae(self, data):
        # Compute RAE for classification, e.g., absolute error
        # Implement the appropriate error metric here
        return 0  # Replace with your implementation
