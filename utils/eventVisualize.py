import numpy as np
import imageio
from PIL import Image


def calEcm(event, shape=[34, 34]):
    if not type(event) is np.ndarray:
        event = np.asarray(event)
    ecm = np.zeros([2, shape[1], shape[0]])
    for e in event:
        x = e[2]
        y = e[1]
        p = e[3]
        ecm[p, x, y] += 1
    return ecm


def drawEcm(ecm, path):
    img = np.zeros((ecm.shape[1], ecm.shape[2], 3))

    # ecm[0] = ecm[0] / ecm[0].max()
    # ecm[1] = ecm[1] / ecm[1].max()

    img[:, :, 1] = 255 * ecm[0]
    img[:, :, 0] = 255 * ecm[1]
    img = img.astype(np.uint8)
    Image.fromarray(img).save(path)
    # imageio.imsave(path, img)


def drawEvent(event, path, shape=[34, 34]):
    if not type(event) is np.ndarray:
        event = np.asarray(event)
    ecm = np.zeros([2, shape[0], shape[1]])
    for e in event:
        x = e[1]
        y = e[2]
        p = e[3]
        ecm[p, x, y] = 1

    img = np.zeros((ecm.shape[1], ecm.shape[2], 3))

    img[:, :, 1] = 255 * ecm[0]
    img[:, :, 0] = 255 * ecm[1]

    imageio.imsave(path, img)


def drawSpike(event, shape, path):
    if not type(event) is np.ndarray:
        event = np.asarray(event)
    img = np.ones([shape[2], shape[0]*shape[1], 3])
    # img = np.zeros([shape[2], shape[0] * shape[1], 3])
    for e in event:
        t = e[0]
        x = e[1]
        y = e[2]
        p = e[3]
        if p == 0:
            img[t, x + shape[1] * y, :] = [0, 0, 1]
        elif p == 1:
            img[t, x + shape[1] * y, :] = [1, 0, 0]

    imageio.imsave(path, img)


if __name__ == '__main__':
    import torch
    # event = np.load('/repository/admin/DVS/Classification/N-MNIST/SR_Test/HR/0/0.npy')
    # drawSpike(event, [34, 34, 350], './spike/t.png')
    #
    # event = np.load('/repository/admin/DVS/Classification/N-MNIST/SR_Test/HRPre/0/0.npy')
    # drawSpike(event, [34, 34, 350], './spike/t1.png')
    #
    # event = np.load('/repository/admin/DVS/Classification/N-MNIST/SR_Test/previousWork1/0/0.npy')
    # drawSpike(event, [34, 34, 350], './spike/t2.png')

    event = np.load('/repository/admin/DVS/Classification/PokerDvs/SR_Test/HR/0/xclub25.npy')
    drawSpike(event, [34, 34, 50], './spike/t.png')
    event = np.load('/repository/admin/DVS/Classification/PokerDvs/SR_Test/HRPre/0/xclub25.npy')
    drawSpike(event, [34, 34, 50], './spike/t1.png')