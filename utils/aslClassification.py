import os
import numpy as np
import sys
import glob

sys.path.append('../')


class Asl:
    def __init__(self, basicPath, train=True):
        if train:
            txtPath = os.path.join("../dataset/ASL/Train.txt")
        else:
            txtPath = os.path.join("../dataset/ASL/Test.txt")

        self.eventList = []
        self.labelList = []
        self.train = train
        if not self.train:
            with open(txtPath, 'r') as f:
                for line in f.readlines():
                    p, label = line.split()
                    print(p, label)
                    event = np.load(os.path.join(basicPath, p.replace('.mat', '.npy'))).astype(np.float32)
                    event = event[:, [1, 2, 0, 3]]
                    if event.shape[0] == 0:
                        continue
                    self.eventList.append(event)
                    self.labelList.append(int(label))
        else:
            with open(txtPath, 'r') as f:
                for line in f.readlines():
                    p, label = line.split()
                    self.eventList.append(os.path.join(basicPath, p.replace('.mat', '.npy')))
                    self.labelList.append(int(label))

    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, idx):
        label = self.labelList[idx]
        if not self.train:
            event = self.eventList[idx]
        else:
            event = np.load(self.eventList[idx]).astype(np.float32)
            event = event[:, [1, 2, 0, 3]]
        return event, label