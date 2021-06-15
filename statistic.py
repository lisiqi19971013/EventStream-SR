

class Metric(object):
    def __init__(self):
        self.reset()

    def updateIter(self, loss_time, loss_ecm, loss_total, num, IS=0, OS=0, GS=0):
        self.LossTime += loss_time
        self.LossEcm += loss_ecm
        self.Num += num
        self.Loss += loss_total
        self.IS += IS
        self.OS += OS
        self.GS += GS

    def reset(self):
        self.LossTime = 0
        self.LossEcm = 0
        self.Loss = 0
        self.Num = 0
        self.IS = 0
        self.OS = 0
        self.GS = 0

    def getAvg(self):
        avgLossTime = self.LossTime / self.Num
        avgLossEcm = self.LossEcm / self.Num
        avgLoss = self.Loss / self.Num
        avgIS = self.IS / self.Num
        avgOS = self.OS / self.Num
        avgGS = self.GS / self.Num

        return avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS
