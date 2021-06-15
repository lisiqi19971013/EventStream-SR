import torch
import numpy as np
from torch.utils.data import Dataset
import os
from slayerSNN.spikeFileIO import event


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    startTime = npEvent[:, 0].min()
    npEvent[:, 0] -= startTime
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3), startTime


class irDataset(Dataset):
    def __init__(self, train=True, shape=[180, 240, 50]):
        self.lrList = []
        self.hrList = []

        self.H = shape[0]
        self.W = shape[1]

        if train:
            self.txt = "../dataset/ImageReconstruction/train.txt"
        else:
            self.txt = "../dataset/ImageReconstruction/test.txt"

        with open(self.txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                lp, hp = line.strip('\n').split(' ')
                self.lrList.append(lp)
                self.hrList.append(hp)

        self.nTimeBins = shape[2]

    def __getitem__(self, idx):
        eventHr, startTime = readNpSpikes(self.hrList[idx])
        eventLr, startTime = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H/2), int(self.W/2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))
        assert eventHr1.sum() == len(eventHr.x)
        assert eventLr1.sum() == len(eventLr.x)
        return eventLr1, eventHr1, startTime

    def __len__(self):
        return len(self.lrList)
