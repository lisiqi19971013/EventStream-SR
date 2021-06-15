import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--savepath', type=str, default='/repository/admin/DVS/Classification/N-MNIST/ckpt_convSNN/full')
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--showFreq', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--cuda', type=str, default='0,1')
parser.add_argument('--add', type=str, default=None)
parser.add_argument('--j', type=int, default=16)
