import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)


class mnistDataset(Dataset):
    def __init__(self, train=True):
        self.lrList = []
        self.hrList = []
        if train:
            self.hrPath = '../dataset/N-MNIST/SR_Train/HR'
            self.lrPath = '../dataset/N-MNIST/SR_Train/LR'
        else:
            self.hrPath = '../dataset/N-MNIST/SR_Test/HR'
            self.lrPath = '../dataset/N-MNIST/SR_Test/LR'

        self.H = 34
        self.W = 34

        for k in range(10):
            print("Read data %d"%k)
            hp = os.path.join(self.hrPath, str(k))
            lp = os.path.join(self.lrPath, str(k))
            assert len(os.listdir(hp)) == len(os.listdir(lp))
            list = os.listdir(hp)

            for n in list:
                self.hrList.append(os.path.join(hp, n))
                self.lrList.append(os.path.join(lp, n))

        self.nTimeBins = 350

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, 17, 17, self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, 34, 34, self.nTimeBins)))

        assert eventHr1.sum() == len(eventHr.x)
        assert eventLr1.sum() == len(eventLr.x)
        return eventLr1, eventHr1

    def __len__(self):
        return len(self.lrList)

