import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        path = join("./trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            print("init kernel")
            self.init_kernel(num_channels)

    def forward(self, x):
        x = x[None, ..., None]

        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            gt_values = self.trilinear_kernel(ts, num_channels)

            values = self.forward(ts)

            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim, mlp_layers=[1, 100, 100, 1], activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers, activation=activation, num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        events = events.float()
        B = int((1+events[-1,-1]).item())
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        x, y, t, p, b = events.t()    # 转置

        for bi in range(B):
            t[events[:, -1] == bi] /= (t[events[:, -1] == bi].max() + 1e-9)

        idx_before_bins = x + W * y + 0 + W * H * C * p + W * H * C * 2 * b

        for i_bin in range(C):
            values = t * self.value_layer.forward(t-i_bin/(C-1))
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)
        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9, 34, 34), num_classes=10, mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1), pretrained=True):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        self.classifier = resnet34(pretrained=pretrained)

        input_channels = 2*voxel_dimension[0]
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

    def forward(self, x):
        vox = self.quantization_layer(x)
        pred = self.classifier(vox)
        return pred


# if __name__ == '__main__':
#     from classification.nmnistDataset import NMnist
#     from classification.classificationDataLoader import Loader
#
#     trainDataset = NMnist(basicPath="/repository/admin/DVS/Classification/N-MNIST/SR_Test/HR/", train=True)
#     trainLoader = Loader(dataset=trainDataset, batch_size=4, device="cuda", num_workers=4, pin_memory=True)
#
#     m = Classifier().cuda()
#     for i, (events, labels) in enumerate(trainLoader):
#         o = m(events)
#         break
