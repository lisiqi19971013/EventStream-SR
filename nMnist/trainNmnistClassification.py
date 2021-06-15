import sys
sys.path.append('..')
from utils.classificationModel import Classifier
from utils.nmnistDatasetClassification import NMnist
from utils.classificationDataLoader import Loader
import datetime
import torch
from utils.ckpt import checkpoint_restore, checkpoint_save
import numpy as np
import os
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy


mode = "HrGt"

trainDataset = NMnist(basicPath="../dataset/N-MNIST/SR_Train/HR/", train=True)
if mode == "HrGt":
    testDataset = NMnist(basicPath="../dataset/N-MNIST/SR_Test/HR/", train=False)
elif mode == "HrPreviousWork":
    testDataset = NMnist(basicPath="../dataset/N-MNIST/SR_Test/previousWork/", train=False)
elif mode == "HrPre":
    testDataset = NMnist(basicPath="../dataset/N-MNIST/Res_Conv/HRPre/", train=False)
elif mode == "Lr":
    trainDataset = NMnist(basicPath="../dataset/N-MNIST/SR_Train/LR/", train=True)
    testDataset = NMnist(basicPath="../dataset/N-MNIST/SR_Test/LR/", train=False)
else:
    raise ValueError

bs = 64
trainLoader = Loader(dataset=trainDataset, device="cuda", batch_size=bs, num_workers=2, pin_memory=True)
testLoader = Loader(dataset=testDataset, device="cuda", batch_size=bs, num_workers=2, pin_memory=True)

shape = (9, 34, 34) if mode != "Lr" else (9, 17, 17)
model = Classifier(voxel_dimension=shape, num_classes=10)
model = model.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
iter_per_epoch = int(trainDataset.__len__() / bs)
time_last = datetime.datetime.now()

savePath = './ckpt_cls/'+mode
model, epoch0 = checkpoint_restore(model, savePath)
print(savePath)
print(trainDataset.path, testDataset.path)

tf_writer = SummaryWriter(log_dir=savePath)

maxEpoch = 30

valAccList = []
trainLossHistory = []
valLossHistory = []
for epoch in range(epoch0+1, maxEpoch):
    trainLoss = []
    trainAccSum = []
    model.train()
    for i, (events, labels) in enumerate(trainLoader):
        pred_labels = model(events)
        loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

        trainLoss.append(loss.item())
        trainAccSum.append(accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % int(iter_per_epoch/10) == 0:
            remainIter = (maxEpoch - epoch - 1) * iter_per_epoch + (iter_per_epoch - i - 1)
            time_now = datetime.datetime.now()
            dt = (time_now - time_last).total_seconds()
            remainSec = remainIter * dt / int(iter_per_epoch/5)
            minute, second = divmod(remainSec, 60)
            hour, minute = divmod(minute, 60)
            t1 = time_now + datetime.timedelta(seconds=remainSec)
            avgLoss = sum(trainLoss) / len(trainLoss)
            avgAcc = sum(trainAccSum) / len(trainAccSum)
            print(mode, 'Cost Time %.04fs, Epoch[%03d]/[%03d], Iter %05d/%05d, Avg Loss: %04f, Acc: %.4f, bs: %d, Remain time: %02d:%02d:%02d, End at:' %
                  (dt, epoch, maxEpoch, i, iter_per_epoch, avgLoss, avgAcc, bs, hour, minute, second), t1.__format__("%Y-%m-%d %H:%M:%S"))
            time_last = time_now

    trainLossHistory.append(sum(trainLoss) / len(trainLoss))
    tf_writer.add_scalar('loss/train', sum(trainLoss) / len(trainLoss), epoch)
    tf_writer.add_scalar('acc/train', sum(trainAccSum) / len(trainAccSum), epoch)


    '''Test'''
    model.eval()
    t = datetime.datetime.now()
    valLoss = []
    valAccSum = []
    for k, (events, labels) in enumerate(testLoader):
        with torch.no_grad():
            pred_labels = model(events)
            loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)
            valLoss.append(loss.item())
            valAccSum.append(accuracy)
    valAvgLoss = sum(valLoss) / len(valLoss)
    valAvgAcc = sum(valAccSum)/len(valAccSum)
    valLossHistory.append(valAvgLoss)
    valAccList.append(valAvgAcc)
    print("Validation Done! Cost Time: %04fs, Loss: %04f, Acc: %.4f, Best Acc: %f, Min Val Loss: %04f" %
          ((datetime.datetime.now()-t).total_seconds(), valAvgLoss, valAvgAcc, max(valAccList), min(valLossHistory)))

    tf_writer.add_scalar('loss/test', valAvgLoss, epoch)
    tf_writer.add_scalar('acc/test', valAvgAcc, epoch)

    if max(valAccList) == valAccList[-1]:
        checkpoint_save(model=model, path=savePath, epoch=epoch, name="ckptBest", device='cuda')

    if epoch % 10 == 9:
        print("Adjusting Learning Rate")
        lr_scheduler.step()

valAccList = np.array(valAccList)
np.save(os.path.join(savePath, 'Max Acc:'+str(max(valAccList).item())+'.npy'), valAccList)


# if __name__ == '__main__':
#     trainClassification("Hr")
    # mode ="Hr"
    # from utils.eventVisualize import drawEcm, calEcm
    # if mode == "Hr":
    #     trainDataset = NMnist(basicPath="/repository/admin/DVS/Classification/N-MNIST/SR_Test/HR/", train=True)
    #     testDataset = NMnist(basicPath="/repository/admin/DVS/Classification/N-MNIST/SR_Test/HR/", train=False)
    # elif mode == "LrGt":
    #     trainDataset = NMnist(basicPath="/repository/admin/DVS/Classification/N-MNIST/SR_Test/LR/", train=True)
    #     testDataset = NMnist(basicPath="/repository/admin/DVS/Classification/N-MNIST/SR_Test/LR/", train=False)
    # elif mode == "LrPre":
    #     trainDataset = NMnist(basicPath="/repository/admin/DVS/Classification/N-MNIST/SR_Test/LRPre/", train=True)
    #     testDataset = NMnist(basicPath="/repository/admin/DVS/Classification/N-MNIST/SR_Test/LRPre/", train=False)
    # else:
    #     raise ValueError
    #
    # bs = 32
    # trainLoader = Loader(dataset=trainDataset, device="cuda", batch_size=bs, num_workers=4, pin_memory=True)
    # testLoader = Loader(dataset=testDataset, device="cuda", batch_size=bs, num_workers=4, pin_memory=True)
    #
    # for i, (events, labels) in enumerate(testLoader):
    #     e = events[events[:,-1] == 24][:,0:4]
    #     x, y, t, p = e.t().detach()
    #     e1 = e.new_full(e.shape, 0)
    #     e1[:,0] = t
    #     e1[:,1] = x
    #     e1[:,2] = y
    #     e1[:,3] = p
    #     ecm = calEcm(e1[1500:2000].int().cpu().detach().numpy())
    #     drawEcm(ecm, './t.png')
    #     break