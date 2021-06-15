import sys
sys.path.append('..')
from model import NetworkBasic
from torch.utils.data import DataLoader, Dataset
import numpy as np
import slayerSNN as snn
import torch
from utils.ckpt import checkpoint_restore
from slayerSNN.spikeFileIO import event
import os
from utils.utils import getEventFromTensor


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    startTime = npEvent[:, 0].min()
    npEvent[:, 0] -= startTime
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3), startTime


ckptPath = './ckpt/'


class irDataset(Dataset):
    def __init__(self, shape=[180, 240, 50]):
        self.lrList = []
        self.hrList = []

        self.H = shape[0]
        self.W = shape[1]

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
        path = self.hrList[idx]
        return eventLr1, eventHr1, startTime, path

    def __len__(self):
        return len(self.lrList)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'
testDataset = irDataset()
bs = 1
testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=False, num_workers=1, drop_last=False)

netParams = snn.params('network.yaml')
m = NetworkBasic(netParams)
m = torch.nn.DataParallel(m).to(device)
print(netParams['simulation'])

m, epoch0 = checkpoint_restore(m, ckptPath, name='ckptBest', device=device)

Mse = torch.nn.MSELoss(reduction='mean')

loss_sum = 0
l = []
count = 0

lossTime = lossEcm = 0
for k, (eventLr, eventHr, startTime, path) in enumerate(testLoader, 0):

    with torch.no_grad():
        eventLr = eventLr.to("cuda")
        eventHr = eventHr.to("cuda")

        output = m(eventLr)

        eventList = getEventFromTensor(output)
        e = eventList[0]
        e = e[:, [0, 2, 1, 3]]
        e[:, 0] = e[:, 0] + startTime[0].item()

        new_path = path[0].replace("HR", "HRPre")
        os.makedirs(os.path.split(new_path)[0], exist_ok=True)

        np.save(new_path, e.astype(np.int32))

    # break
    if k % 100 == 99:
        print(k, '/', len(testDataset), lossTime/(k+1), lossEcm/(k+1))
