import sys
sys.path.append('..')
from model import NetworkBasic, Network1, Network2, Network3
from torch.utils.data import DataLoader, Dataset
import datetime, os
import slayerSNN as snn
import torch
from utils.utils import getEventFromTensor
from utils.ckpt import checkpoint_restore
import numpy as np
from slayerSNN.spikeFileIO import event


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'


def readNpSpikes(filename, timeUnit=1e-3):
    npEvent = np.load(filename)
    return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)


savepath = '../dataset/N-MNIST/ResConv/HRPre'
ckptPath = './ckpt/'


class mnistDataset(Dataset):
    def __init__(self):
        self.lrList = []
        self.hrList = []
        self.hrPath = '../dataset/N-MNIST/SR_Test/HR'
        self.lrPath = '../dataset/N-MNIST/SR_Test/LR'
        self.path = []

        self.H = 34
        self.W = 34

        for k in range(10):
            print("Read data %d"%k)
            hp = os.path.join(self.hrPath, str(k))
            lp = os.path.join(self.lrPath, str(k))
            if not os.path.exists(os.path.join(savepath, str(k))):
                os.makedirs(os.path.join(savepath, str(k)))
            assert len(os.listdir(hp)) == len(os.listdir(lp))
            list = os.listdir(hp)
            for n in list:
                self.hrList.append(os.path.join(hp, n))
                self.lrList.append(os.path.join(lp, n))
                self.path.append(os.path.join(str(k), n))

        self.nTimeBins = 350

    def __getitem__(self, idx):
        eventHr = readNpSpikes(self.hrList[idx])
        eventLr = readNpSpikes(self.lrList[idx])

        eventLr1 = eventLr.toSpikeTensor(torch.zeros((2, 17, 17, self.nTimeBins)))
        eventHr1 = eventHr.toSpikeTensor(torch.zeros((2, 34, 34, self.nTimeBins)))

        assert eventHr1.sum() == len(eventHr.x)
        assert eventLr1.sum() == len(eventLr.x)

        path = self.path[idx]
        return eventLr1, eventHr1, path

    def __len__(self):
        return len(self.lrList)


testDataset = mnistDataset()
with open(os.path.join(savepath, 'ckpt.txt'), 'w') as f:
    f.writelines(ckptPath)

bs = 1
testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=False, num_workers=0)

netParams = snn.params('network.yaml')
m = NetworkBasic(netParams).to("cuda")
m = torch.nn.DataParallel(m).to(device)
m.eval()

m, epoch0 = checkpoint_restore(m, ckptPath, name='ckptBest', device=device)
print("start from epoch %d" % epoch0)

Mse = torch.nn.MSELoss(reduction='mean')

loss_sum = 0
l = []
count = 0

lossTime = lossEcm = 0

for k, (eventLr, eventHr, path) in enumerate(testLoader):
    with torch.no_grad():
        eventLr = eventLr.to("cuda")
        eventHr = eventHr.to("cuda")

        output = m(eventLr)

        eventList = getEventFromTensor(output)
        e = eventList[0]
        e = e[:, [0, 2, 1, 3]]
        new_path = os.path.join(savepath, path[0])
        np.save(new_path, e.astype(np.int32))

        if k % 100 ==0:
            print("%d/%d"%(k, len(testLoader)))
