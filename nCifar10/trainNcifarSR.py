import sys
sys.path.append('../')
from model import NetworkBasic
from nCifar10.ncifarDatasetSR import ncifarDataset
from torch.utils.data import DataLoader
import datetime
import slayerSNN as snn
import torch
from utils.ckpt import checkpoint_restore, checkpoint_save
import os
from opts import parser
from statistic import Metric
from tensorboardX import SummaryWriter

torch.backends.cudnn.enabled = False

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
device = 'cuda'

shape = [128, 128, 1500]
trainDataset = ncifarDataset(train=True, shape=shape)
testDataset = ncifarDataset(train=False, shape=shape)

print("Training sample: %d, Testing sample: %d" % (trainDataset.__len__(), testDataset.__len__()))
bs = args.bs

trainLoader = DataLoader(dataset=trainDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=True)
testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=False)

netParams = snn.params('network.yaml')
m = NetworkBasic(netParams)
m = torch.nn.DataParallel(m).to(device)
print(m)

MSE = torch.nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=args.lr, amsgrad=True)

iter_per_epoch = int(trainDataset.__len__() / bs)
time_last = datetime.datetime.now()

savePath = args.savepath
print(savePath)
m, epoch0 = checkpoint_restore(m, savePath, name='ckpt')

maxEpoch = args.epoch
showFreq = args.showFreq
valLossHistory = []
tf_writer = SummaryWriter(log_dir=savePath)
with open(os.path.join(savePath, 'config.txt'), 'w') as f:
    for i, config in enumerate(m.module.neuron_config):
        f.writelines('layer%d: theta=%d, tauSr=%.2f, tauRef=%.2f, scaleRef=%.2f, tauRho=%.2f, scaleRho=%.2f\n' % (
            i + 1, config['theta'], config['tauSr'], config['tauRef'], config['scaleRef'], config['tauRho'],
            config['scaleRho']))
    f.writelines('\n')
    f.write(str(args))

log_training = open(os.path.join(savePath, 'log.csv'), 'w')

for epoch in range(epoch0+1, maxEpoch):
    trainMetirc = Metric()
    m.train()
    for i, (eventLr, eventHr) in enumerate(trainLoader, 0):

        num = eventLr.shape[0]
        eventLr = eventLr.to(device)
        eventHr = eventHr.to(device)
        output = m(eventLr)

        loss = MSE(output, eventHr)
        loss_ecm = MSE(torch.sum(output[:, :, :, :, 0:50], dim=4), torch.sum(eventHr[:, :, :, :, 0:50], dim=4))
        for k in range(1, int(shape[2]/50)):
            loss_ecm += MSE(torch.sum(output[:, :, :, :, 50 * k:50 * k + 50], dim=4), torch.sum(eventHr[:, :, :, :, 50 * k:50 * k + 50], dim=4))
        loss_total = loss + loss_ecm * 5

        '''
        Warning:
            We have tested to use the spikeTime Loss provided in slayerSNN.spikeLoss to calculate the temporal loss,
            and use the numSpikes Loss provided in slayerSNN.spikeLoss to calculate the spatial loss.

            It achieves the same performance as the MSE loss,  but the calculations are extremely slow (about 10 times).
        '''

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if (i) % showFreq == 0:
            trainMetirc.updateIter(loss.item(), loss_ecm.item(), loss_total.item(), 1,
                                   eventLr.sum().item(), output.sum().item(), eventHr.sum().item())
            remainIter = (maxEpoch - epoch -1) * iter_per_epoch + (iter_per_epoch - i - 1)
            time_now = datetime.datetime.now()
            dt = (time_now - time_last).total_seconds()
            remainSec = remainIter * dt / showFreq
            minute, second = divmod(remainSec, 60)
            hour, minute = divmod(minute, 60)
            t1 = time_now + datetime.timedelta(seconds=remainSec)

            avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = trainMetirc.getAvg()
            message = 'Train, Cost %.1fs, Epoch[%d]/[%d], Iter %d/%d, Time Loss: %f, Ecm Loss: %f, Avg Loss: %f, ' \
                     'bs: %d, IS: %d, OS: %d, GS: %d, Remain time: %02d:%02d:%02d, End at:' % \
                     (dt, epoch, maxEpoch, i, iter_per_epoch, avgLossTime, avgLossEcm, avgLoss, bs, avgIS, avgOS, avgGS,
                      hour, minute, second) + t1.__format__("%Y-%m-%d %H:%M:%S")
            print(message)
            if log_training is not None:
                log_training.write(message + '\n')
                log_training.flush()
            time_last = time_now

    avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = trainMetirc.getAvg()
    tf_writer.add_scalar('loss/Train_Time_Loss', avgLossTime, epoch)
    tf_writer.add_scalar('loss/Train_Spatial_Loss', avgLossEcm, epoch)
    tf_writer.add_scalar('loss/Train_Total_Loss', avgLoss, epoch)
    tf_writer.add_scalar('SpikeNum/Train_Input', avgIS, epoch)
    tf_writer.add_scalar('SpikeNum/Train_Output', avgOS, epoch)
    tf_writer.add_scalar('SpikeNum/Train_GT', avgGS, epoch)

    message = '-' * 50 + "Epoch %d Done" % epoch + '-' * 50
    print(message)
    if log_training is not None:
        log_training.write(message + '\n')
        log_training.flush()

    if epoch % 1 == 0:
        m.eval()
        t = datetime.datetime.now()
        valMetirc = Metric()
        for i, (eventLr, eventHr) in enumerate(testLoader, 0):
            with torch.no_grad():
                num = eventLr.shape[0]
                eventLr = eventLr.to(device)
                eventHr = eventHr.to(device)
                output = m(eventLr)

                loss = MSE(output, eventHr)
                loss_ecm = MSE(torch.sum(output[:, :, :, :, 0:50], dim=4), torch.sum(eventHr[:, :, :, :, 0:50], dim=4))
                for k in range(1, int(shape[2] / 50)):
                    loss_ecm += MSE(torch.sum(output[:, :, :, :, 50 * k:50 * k + 50], dim=4), torch.sum(eventHr[:, :, :, :, 50 * k:50 * k + 50], dim=4))
                loss_total = loss + loss_ecm
                valMetirc.updateIter(loss.item(), loss_ecm.item(), loss_total.item(), 1,
                                     eventLr.sum().item(), output.sum().item(), eventHr.sum().item())

                if (i) % showFreq == 0:
                    remainIter = (maxEpoch - epoch - 1) * iter_per_epoch + (iter_per_epoch - i - 1)
                    time_now = datetime.datetime.now()
                    dt = (time_now - time_last).total_seconds()
                    remainSec = remainIter * dt / showFreq
                    minute, second = divmod(remainSec, 60)
                    hour, minute = divmod(minute, 60)
                    t1 = time_now + datetime.timedelta(seconds=remainSec)

                    avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = valMetirc.getAvg()
                    message = 'Val, Cost %.1fs, Epoch[%d], Iter %d/%d, Time Loss: %f, Ecm Loss: %f, Avg Loss: %f,' \
                             ' IS: %d, OS: %d, GS: %d, Remain time: %02d:%02d:%02d, End at:' % \
                             (dt, epoch, i, len(testDataset)/args.bs, avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS,
                             hour, minute, second) + t1.__format__("%Y-%m-%d %H:%M:%S")
                    print(message)
                    if log_training is not None:
                        log_training.write(message + '\n')
                        log_training.flush()
                    time_last = time_now

        avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = valMetirc.getAvg()
        tf_writer.add_scalar('loss/Val_Time_Loss', avgLossTime, epoch)
        tf_writer.add_scalar('loss/Val_Spatial_Loss', avgLossEcm, epoch)
        tf_writer.add_scalar('loss/Val_Total_Loss', avgLoss, epoch)
        tf_writer.add_scalar('SpikeNum/Val_Input', avgIS, epoch)
        tf_writer.add_scalar('SpikeNum/Val_Output', avgOS, epoch)
        tf_writer.add_scalar('SpikeNum/Val_GT', avgGS, epoch)

        valLossHistory.append(avgLoss)
        time_last = datetime.datetime.now()
        message = "Validation Done! Cost Time: %.2fs, Loss Time: %f, Loss Ecm: %f, Avg Loss: %f, Min Val Loss: %f\n" %\
                  ((time_last-t).total_seconds(), avgLossTime, avgLossEcm, avgLoss, min(valLossHistory))
        print(message)
        if log_training is not None:
            log_training.write(message + '\n')
            log_training.flush()

        checkpoint_save(model=m, path=savePath, epoch=epoch, name="ckpt", device=device)

        if (min(valLossHistory) == valLossHistory[-1]):
            checkpoint_save(model=m, path=savePath, epoch=epoch, name="ckptBest", device=device)

        with open(os.path.join(savePath, 'log.txt'), "a") as f:
            f.write("Epoch: %d, Ecm loss: %f, Spike time loss: %f, Total loss: %f\n" %
                    (epoch, avgLossEcm, avgLossTime, avgLoss))

    if (epoch+1) % 15 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            print(param_group['lr'])
