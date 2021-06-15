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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

savepath = '../dataset/Cifar10-DVS/Res_Conv/HRPre'
ckptPath = './ckpt/'


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)


class cifarDataset(Dataset):
    def __init__(self):
        self.lrList = []
        self.hrList = []
        self.hrPath = '../dataset/Cifar10-DVS/SR_Test/HR'
        self.lrPath = '../dataset/Cifar10-DVS/SR_Test/LR'
        classList = ['frog', 'dog', 'deer', 'ship', 'bird', 'horse', 'airplane', 'automobile', 'cat', 'truck']
        self.H = 128
        self.W = 128

        self.path = []
        for k in classList:
            hp = os.path.join(self.hrPath, k)
            lp = os.path.join(self.lrPath, k)
            if not os.path.exists(os.path.join(savepath, k)):
                os.makedirs(os.path.join(savepath, k))
            assert len(os.listdir(hp)) == len(os.listdir(lp))
            list = os.listdir(hp)
            print("Read data ", k, len(os.listdir(lp)))

            for c, n in enumerate(list):
                self.hrList.append(os.path.join(hp, n))
                self.lrList.append(os.path.join(lp, n))
                self.path.append(os.path.join(str(k), n))
        self.nTimeBins = 1500

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, int(self.H/2), int(self.W/2), self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, self.H, self.W, self.nTimeBins)))

        assert eventHr1.sum() == len(eventHr.x)
        assert eventLr1.sum() == len(eventLr.x)
        path = self.path[idx]
        return eventLr1, eventHr1, path

    def __len__(self):
        return len(self.lrList)


device = 'cuda'
testDataset = cifarDataset()
bs = 1
testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=False, num_workers=1, drop_last=False)

netParams = snn.params('network.yaml')
m = NetworkBasic(netParams)
m = torch.nn.DataParallel(m).to(device)
m.eval()

print(netParams['simulation'])

m, epoch0 = checkpoint_restore(m, ckptPath, name="ckptBest")


for k, (eventLr, eventHr, path) in enumerate(testLoader, 0):
    with torch.no_grad():
        eventLr = eventLr.to("cuda")
        eventHr = eventHr.to("cuda")

        output = m(eventLr)

        eventList = getEventFromTensor(output)
        e = eventList[0]
        e = e[:, [0,2,1,3]]
        new_path = os.path.join(savepath, path[0])
        np.save(new_path, e.astype(np.int32))

    if k % 100 == 0:
        print(k, '/', len(testDataset))

