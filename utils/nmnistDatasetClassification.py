import os
import numpy as np


class NMnist:
    def __init__(self, basicPath, train=True):
        if train:
            txtPath = os.path.join("../dataset/N-MNIST/TrainList.txt")
        else:
            txtPath = os.path.join("../dataset/N-MNIST/TestList.txt")
        self.path = basicPath
        self.eventList = []
        self.labelList = []
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                p, label = line.split()
                self.eventList.append(os.path.join(basicPath, p))
                self.labelList.append(int(label))

    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, idx):
        label = self.labelList[idx]
        f = self.eventList[idx]
        event = np.load(f).astype(np.float32)
        event = event[:, [1, 2, 0, 3]]
        event = event[event[:,2].argsort()]
        return event, label
