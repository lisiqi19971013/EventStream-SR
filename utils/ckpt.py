import torch
import os


def checkpoint_restore(model, path, name="ckpt", device='cuda'):
    model.cpu()
    f = os.path.join(path, name+'.pth')
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(f):
        print('Loaded ' + f)
        model_CKPT = torch.load(f)
        model.load_state_dict(model_CKPT['state_dic'])
        if 'epoch' in model_CKPT:
            epoch = model_CKPT['epoch']
            print("Load Epoch %d!!!"%epoch)
        else:
            epoch = -1
    else:
        print(f)
        print("Start New Training")
        epoch = -1

    model.to(device)
    return model, epoch


def checkpoint_save(model, path, epoch, name="ckpt", device='cuda'):
    if not os.path.exists(path):
        os.makedirs(path)
    if name == "ckptBest":
        f = os.path.join(path, name + '.pth')
    else:
        f = os.path.join(path, name + str(epoch) + '.pth')
    model.cpu()
    torch.save({'state_dic': model.state_dict(), "epoch": epoch}, f)
    model.to(device)