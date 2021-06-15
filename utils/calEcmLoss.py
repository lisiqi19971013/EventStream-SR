import torch
import numpy as np
from slayerSNN.slayer import spikeLayer


class EcmLoss(torch.nn.Module):
    def __init__(self, networkDescriptor, slayerClass=spikeLayer):
        super(EcmLoss, self).__init__()
        self.neuron = networkDescriptor['neuron']
        self.simulation = networkDescriptor['simulation']
        self.slayer = slayerClass(self.neuron, self.simulation)

    def calEcmLoss(self, spikeOut, spikeDesire, begin, end):

        actualSpikes = torch.sum(spikeOut[..., begin:end], 4, keepdim=True).cpu().detach().numpy()
        desiredSpikes = torch.sum(spikeDesire[..., begin:end], 4, keepdim=True).cpu().detach().numpy()

        errorSpikeCount = (actualSpikes - desiredSpikes)
        targetRegion = np.zeros(spikeOut.shape)
        targetRegion[:, :, :, :, begin:end] = 1;
        spikeDesired = torch.FloatTensor(targetRegion * spikeOut.cpu().data.numpy()).to(spikeOut.device)

        error = self.slayer.psp(spikeOut - spikeDesired)
        error += torch.FloatTensor(errorSpikeCount * targetRegion).to(spikeOut.device)

        return 1 / 2 * torch.sum(error ** 2)
