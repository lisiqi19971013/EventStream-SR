import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event
import random


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)


class aslDataset(Dataset):
    def __init__(self, train=True, shape=[180, 240, 200]):
        self.lrList = []
        self.hrList = []
        self.train = train
        self.H = shape[0]
        self.W = shape[1]

        if train:
            self.hrPath = '../dataset/ASL/SR_Train/HR'
            self.lrPath = '../dataset/ASL/SR_Train/LR'
        else:
            self.hrPath = '../dataset/ASL/SR_Test/HR'
            self.lrPath = '../dataset/ASL/SR_Test/LR'

        classList = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

        for k in classList:
            hp = os.path.join(self.hrPath, k)
            lp = os.path.join(self.lrPath, k)
            assert len(os.listdir(hp)) == len(os.listdir(lp))
            list = os.listdir(hp)
            for n in list:
                self.hrList.append(os.path.join(hp, n))
                self.lrList.append(os.path.join(lp, n))

        self.nTimeBins = shape[2]

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H/2), int(self.W/2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

        return eventLr1, eventHr1

    def __len__(self):
        return 100
        # return len(self.lrList)