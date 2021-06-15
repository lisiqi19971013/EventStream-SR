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


dataset = ["dynamic_6dof", 'boxes_6dof', 'poster_6dof', 'shapes_6dof', 'office_zigzag', 'slider_depth', 'calibration']
beginList = [5, 5, 5, 5, 5, 1, 5]
endList = [20, 20, 20, 20, 12, 2.5, 20]

path = "/repository/admin/DVS/ImageReconstruction/test/"
_H, _W, _T = [240, 180, 50]


with open(os.path.join(path, 'RMSE_Ours.txt'), 'w') as f:
    for k in range(7):
        d = dataset[k]
        p1 = os.path.join(path, d, "SR_Test", 'HRPre')  # Output
        p2 = os.path.join(path, d, "SR_Test", 'HR')  # Gt
        begin = beginList[k] * 1000
        end = endList[k] * 1000

        sampleList = os.listdir(p1)
        RMSEListOurs, RMSEListOurs_s, RMSEListOurs_t = [], [], []

        for name in sampleList:
            eventGt = np.load(os.path.join(p2, name))
            t0 = eventGt[0, 0]
            if t0 < begin or t0 >= end:
                print(t0)
                continue

            eventOutput = torch.from_numpy(np.load(os.path.join(p1, name))).to("cuda")
            eventGt = torch.from_numpy(eventGt).to("cuda")
            assert abs(eventOutput[0, 0] - eventGt[0, 0]) < 25

            eventOutput[:, 0] = eventOutput[:, 0] - t0
            eventGt[:, 0] = eventGt[:, 0] - t0

            RMSE1, RMSE_t, RMSE_s = calRMSE(eventOutput, eventGt)

            RMSEListOurs.append(RMSE1)
            print(d, name, t0, RMSE1)

        print(d, sum(RMSEListOurs)/len(RMSEListOurs))

        f.writelines(d + " RMSE: " + str(sum(RMSEListOurs)/len(RMSEListOurs)) + '\n' )
