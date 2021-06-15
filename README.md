# EventStream-SR
This repository is for the ICCV 2021 anonymous submission: ___Event Stream Super-Resolution via Spatiotemporal Constraint Learning___. This is the raw, unintegrated code.



## Requirements

1. Python 3 with the following packages installed:
   * torch==1.3.1
   * torchvision==0.4.2
   * tqdm==4.61.1
   * numpy==1.19.2
   * imageio==2.9.0
   * Pillow==8.2.0
   * tensorboardX==2.2
   * pyyaml==5.4.1
2. slayerPytorch
   - See https://github.com/bamsumit/slayerPytorch to install the slayerPytorch for the SNN simulation.

3. cuda
   - A **CUDA** enabled **GPU** is required for training any model. We test our code with CUDA 10.0 V10.0.130 and cudnn 7.6.5.



## Data preparing

1. Our datasets is available at https://drive.google.com/drive/folders/1l1v6eqBxRXbRFa6yNphfTy5eVbi_myck?usp=sharing
2. Download the datasets and unzip them to the folder (./dataset/)
3. Change the corresponding data path in each .py file



## Train and test

### Training

This is the raw and unintegrated code. If you want to train the model on a dataset, such as N-MNIST, run this code

```python
>>> cd nMnist
>>> python trainNmnist.py --bs 64 --savepath './ckpt/' --epoch 30 -- showFreq 50  --lr 0.1 -- cuda '1' --j 4
or
>>> sh train.sh
```

### Testing

After training, run the following code to generating results. NOTICE: the default output path is (./dataset/N-MNIST/ResConv/HRPre)

```
>>> python testNmnist.py
```

