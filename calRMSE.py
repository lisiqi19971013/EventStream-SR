import numpy as np
import os
import math
import torch


def calRMSE(eventOutput, eventGt):
    xOp = eventOutput[:, 1].long()
    yOp = eventOutput[:, 2].long()
    pOp = eventOutput[:, 3].long()
    tOp = eventOutput[:, 0].long()

    xGt = eventGt[:, 1].long()
    yGt = eventGt[:, 2].long()
    pGt = eventGt[:, 3].long()
    tGt = eventGt[:, 0].long()

    VoxOp = torch.zeros([2, _H, _W, _T]).to("cuda")
    VoxOp[pOp, xOp, yOp, tOp] = 1
    VoxGt = torch.zeros([2, _H, _W, _T]).to("cuda")
    VoxGt[pGt, xGt, yGt, tGt] = 1

    ecm = torch.sum(torch.sum(VoxGt, dim=3), dim=0)
    assert ecm.sum() == xGt.shape[0]

    RMSE1 = torch.sum( (VoxGt - VoxOp) * (VoxGt - VoxOp) )
    RMSE2 = 0
    for k in range(math.ceil(_T/50)):
        psthGt = torch.sum(VoxGt[:, :, :, k*50:(k+1)*50], dim=3)
        psthOp = torch.sum(VoxOp[:, :, :, k*50:(k+1)*50], dim=3)
        RMSE2 += torch.sum( (psthGt - psthOp) * (psthGt - psthOp) )

    RMSE = torch.sqrt( (RMSE1 + RMSE2) / ( (tGt.max()-tGt.min()) * torch.sum(ecm!=0) ))

    return RMSE.item(), \
           (torch.sqrt( (RMSE1) / ( (tGt.max()-tGt.min()) * torch.sum(ecm!=0) ))).item(), \
           (torch.sqrt( (RMSE2) / ( (tGt.max()-tGt.min()) * torch.sum(ecm!=0) ))).item()



path = "./dataset/N-MNIST/Res_Conv/"
_H, _W, _T = [34, 34, 350]

# path = "./dataset/Cifar10-DVS/Res_Conv"
# _H, _W, _T = [128, 128, 1500]

# path = "./dataset/ASL/Res_Conv"
# _H, _W, _T = [240, 180, 600]

classList = os.listdir(os.path.join(path, 'HR'))

RMSEListOurs, RMSEListOurs_s, RMSEListOurs_t = [], [], []
RMSEListPrevious, RMSEListPrevious_s, RMSEListPrevious_t = [], [], []
RMSEListNoEcmLoss, RMSEListNoEcmLoss_s, RMSEListNoEcmLoss_t = [], [], []
RMSEListNoTimeLoss, RMSEListNoTimeLoss_s, RMSEListNoTimeLoss_t = [], [], []

i = 1
for n in classList:
    print(n)
    p1 = os.path.join(path, 'HRPre', n)              # Output
    p2 = os.path.join(path, 'HR', n)                 # Gt
    # p3 = os.path.join(path, 'previousWork', n)      # Previous Work
    # p4 = os.path.join(path, 'HRPre no ecmLoss', n)   # no ecmLoss
    # p5 = os.path.join(path, 'HRPre no timeLoss', n)  # no timeLoss
    # print(p3)

    k = 1
    sampleList = os.listdir(p2)

    for name in sampleList:
        eventOutput = torch.from_numpy(np.load(os.path.join(p1, name))).to("cuda")
        eventGt = torch.from_numpy(np.load(os.path.join(p2, name))).to("cuda")
        # eventPrevious = torch.from_numpy(np.load(os.path.join(p3, name))).to("cuda")
        # eventNoEcmLoss = torch.from_numpy(np.load(os.path.join(p4, name))).to("cuda")
        # eventNoTimeLoss = torch.from_numpy(np.load(os.path.join(p5, name))).to("cuda")

        RMSE1, RMSE_t, RMSE_s = calRMSE(eventOutput, eventGt)
        RMSEListOurs.append(RMSE1)
        RMSEListOurs_s.append(RMSE_s)
        RMSEListOurs_t.append(RMSE_t)

        # RMSE2, RMSE_t, RMSE_s = calRMSE(eventPrevious, eventGt)
        # RMSEListPrevious.append(RMSE2)
        # RMSEListPrevious_s.append(RMSE_s)
        # RMSEListPrevious_t.append(RMSE_t)

        # RMSE3, RMSE_t, RMSE_s = calRMSE(eventNoEcmLoss, eventGt)
        # RMSEListNoEcmLoss.append(RMSE3)
        # RMSEListNoEcmLoss_s.append(RMSE_s)
        # RMSEListNoEcmLoss_t.append(RMSE_t)

        # RMSE4, RMSE_t, RMSE_s = calRMSE(eventNoTimeLoss, eventGt)
        # RMSEListNoTimeLoss.append(RMSE4)
        # RMSEListNoTimeLoss_s.append(RMSE_s)
        # RMSEListNoTimeLoss_t.append(RMSE_t)

        # print(i, '/', len(classList), '  ', k, '/', len(sampleList), RMSE1, RMSE2, RMSE3, RMSE4)
        print(i, '/', len(classList), '  ', k, '/', len(sampleList), RMSE1)
        k += 1
    i += 1

# print(sum(RMSEListOurs) / len(RMSEListOurs), sum(RMSEListPrevious) / len(RMSEListPrevious),
#        sum(RMSEListNoEcmLoss) / len(RMSEListNoEcmLoss), sum(RMSEListNoTimeLoss) / len(RMSEListNoTimeLoss))

print(sum(RMSEListOurs) / len(RMSEListOurs))


with open(path + '/results.txt', 'w') as f:
    f.writelines(p1 + '\n')
    # f.writelines(p2 + '\n')
    # f.writelines(p3 + '\n')
    # f.writelines(p4 + '\n')
    # f.writelines(p5 + '\n')
    f.writelines('Ours RMSE: ' + str(sum(RMSEListOurs)/len(RMSEListOurs)) + ', Ours RMSE_s: ' + str(sum(RMSEListOurs_s)/ len(RMSEListOurs)) + ', Ours RMSE_t: ' + str(sum(RMSEListOurs_t)/ len(RMSEListOurs)) + '\n')
    # f.writelines('Previous RMSE: ' + str(sum(RMSEListPrevious)/ len(RMSEListPrevious))  + ', Previous RMSE_s: ' +str(sum(RMSEListPrevious_s)/ len(RMSEListPrevious)) + ', Previous RMSE_t: '+str(sum(RMSEListPrevious_t)/ len(RMSEListPrevious)) +'\n')
    # f.writelines('NoEcmLoss RMSE: '+ str(sum(RMSEListNoEcmLoss)/len(RMSEListNoEcmLoss)) + ', NoEcmLoss RMSE_s: '+str(sum(RMSEListNoEcmLoss_s)/len(RMSEListNoEcmLoss))+', NoEcmLoss RMSE_t: '+str(sum(RMSEListNoEcmLoss_t)/len(RMSEListNoEcmLoss))+ '\n')
    # f.writelines('NoTimeLoss RMSE: '+str(sum(RMSEListNoTimeLoss)/len(RMSEListNoTimeLoss))+', NoTimeLoss RMSE_s: '+str(sum(RMSEListNoTimeLoss_s)/len(RMSEListNoTimeLoss))+', NoTimeLoss RMSE_t: '+str(sum(RMSEListNoTimeLoss_t)/len(RMSEListNoTimeLoss))+ '\n')