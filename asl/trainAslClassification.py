import sys
sys.path.append('../')
from utils.classificationModel import Classifier
from utils.aslClassification import Asl
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

trainDataset = Asl(basicPath="../dataset/ASL/SR_Train/HR/", train=True)   # 240 180
if mode == "HrGt":
    testDataset = Asl(basicPath="../dataset/ASL/SR_Test/HR/", train=False)
elif mode == "HrPre":
    testDataset = Asl(basicPath="../dataset/ASL/Res_Conv/HrPre/", train=False)
elif mode == "previousWork":
    testDataset = Asl(basicPath="../dataset/ASL/SR_Test/previousWork/", train=False)
elif mode == "Lr":
    trainDataset = Asl(basicPath="../dataset/ASL/SR_Train/LR/", train=True)
    testDataset = Asl(basicPath="../dataset/ASL/SR_Test/LR/", train=False)
else:
    raise ValueError

bs = 16
trainLoader = Loader(dataset=trainDataset, device="cuda", batch_size=bs, num_workers=2, pin_memory=True)
testLoader = Loader(dataset=testDataset, device="cuda", batch_size=bs, num_workers=2, pin_memory=True)
print(trainDataset.__len__(), testDataset.__len__())

shape = (9, 180, 240) if mode != "Lr" else (9, 90, 120)
model = Classifier(voxel_dimension=shape, num_classes=24)

model.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
iter_per_epoch = int(trainDataset.__len__() / bs)
time_last = datetime.datetime.now()

savePath = './ckpt_cls/'+mode
model, epoch0 = checkpoint_restore(model, savePath)
tf_writer = SummaryWriter(log_dir=savePath)

maxEpoch = 30

valAccList = []
trainLossHistory = []
valLossHistory = []
for epoch in range(epoch0+1, maxEpoch):
    valLoss = []
    valAccSum = []
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

        torch.cuda.empty_cache()

        if (i) % int(iter_per_epoch/10) == 0:
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
    t = datetime.datetime.now()
    model.eval()
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

    if max(valAccList) == valAccList[-1]:
        checkpoint_save(model=model, path=savePath, epoch=29, name="ckptBest", device='cuda')
        print('save done!')

    checkpoint_save(model=model, path=savePath, epoch=epoch, name="ckptBest", device='cuda')

    tf_writer.add_scalar('loss/test', valAvgLoss, epoch)
    tf_writer.add_scalar('acc/test', valAvgAcc, epoch)

    with open(os.path.join(savePath, 'log.txt'), "a") as f:
        f.write("Epoch: %d, Acc: %f\n" % (epoch, valAvgAcc))

    if epoch % 10 == 9:
        print("Adjusting Learning Rate")
        lr_scheduler.step()

valAccList = np.array(valAccList)
np.save(savePath+'Max Acc:'+str(max(valAccList).item())+'.npy', valAccList)