import numpy as np
import os
import math


def calRMSE(eventOutput, eventGt):
    xOp = np.round(eventOutput[:, 1]).astype(int)
    yOp = np.round(eventOutput[:, 2]).astype(int)
    pOp = np.round(eventOutput[:, 3]).astype(int)
    tOp = np.round(eventOutput[:, 0]).astype(int)

    xGt = np.round(eventGt[:, 1]).astype(int)
    yGt = np.round(eventGt[:, 2]).astype(int)
    pGt = np.round(eventGt[:, 3]).astype(int)
    tGt = np.round(eventGt[:, 0]).astype(int)

    VoxOp = np.zeros([2, _H, _W, _T])
    VoxOp[pOp, xOp, yOp, tOp] = 1
    VoxGt = np.zeros([2, _H, _W, _T])
    VoxGt[pGt, xGt, yGt, tGt] = 1
    ecm = np.sum(np.sum(VoxGt, axis=3), axis=0)
    assert ecm.sum() == xGt.shape[0]

    RMSE1 = np.sum( (VoxGt - VoxOp) * (VoxGt - VoxOp) )
    RMSE2 = 0
    for k in range(math.ceil(_T/50)):
        psthGt = np.sum(VoxGt[:, :, :, k*50:(k+1)*50], axis=3)
        psthOp = np.sum(VoxOp[:, :, :, k*50:(k+1)*50], axis=3)
        RMSE2 += np.sum( (psthGt - psthOp) * (psthGt - psthOp) )

    RMSE = np.sqrt( (RMSE1 + RMSE2) / ( (tGt.max()-tGt.min()) * np.sum(ecm!=0) ))

    return RMSE, np.sqrt( (RMSE1) / ( (tGt.max()-tGt.min()) * np.sum(ecm!=0) )), np.sqrt( (RMSE2) / ( (tGt.max()-tGt.min()) * np.sum(ecm!=0) ))



# path = "./dataset/N-MNIST/SR_Test"
# path1 = "./dataset/N-MNIST/SR_Test"
# _H, _W, _T = [34, 34, 350]

# path = "./dataset/Cifar10-DVS/SR_Test"
# path1 = "./dataset/Cifar10-DVS/ResConv/HRPre"
# _H, _W, _T = [128, 128, 1500]

# path = "./dataset/asl/SR_Test"
# path1 = "./dataset/asl/ResConv/HRPre"
# _H, _W, _T = [240, 180, 600]

path = "./dataset/ImageReconstruction/SR_Test"
path1 = "./dataset/ImageReconstruction/ResConv/HRPre"
_H, _W, _T = [240, 180, 600]

classList = os.listdir(os.path.join(path, 'HR'))

RMSEListOurs, RMSEListOurs_s, RMSEListOurs_t = [], [], []


i = 1
for n in classList:
    print(n)
    p1 = os.path.join(path, 'HRPre', n)              # Output
    p2 = os.path.join(path, 'HR', n)                 # Gt

    k = 1
    sampleList = os.listdir(p2)

    for name in sampleList:
        eventOutput = np.load(os.path.join(p1, name))
        eventGt = np.load(os.path.join(p2, name))

        RMSE, RMSE_t, RMSE_s = calRMSE(eventOutput, eventGt)
        RMSEListOurs.append(RMSE)
        RMSEListOurs_s.append(RMSE_s)
        RMSEListOurs_t.append(RMSE_t)

        print(i, '/', len(classList), '  ', k, '/', len(sampleList), RMSE)
        k += 1
    i += 1

print(sum(RMSEListOurs) / len(RMSEListOurs))


with open(path1 + '/result.txt', 'w') as f:
    f.writelines('Ours RMSE: ' + str(sum(RMSEListOurs)/len(RMSEListOurs)) + ', Ours RMSE_s: ' + str(sum(RMSEListOurs_s)/ len(RMSEListOurs)) + ', Ours RMSE_t: ' + str(sum(RMSEListOurs_t)/ len(RMSEListOurs)) + '\n')