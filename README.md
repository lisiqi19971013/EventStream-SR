# EventStream-SR
This repository is for the ICCV 2021 anonymous submission: ___Event Stream Super-Resolution via Spatiotemporal Constraint Learning___. 

This is our raw, unintegrated code.



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

1. To ensure anonymity, we shared our data with anonymous accounts called *iccvSubmission*
   -  N-MNIST: https://drive.google.com/file/d/19VNS5gJBHyKKCzsyg9OlBrw3o1dx5UM2/view?usp=sharing
   - Cifar10-DVS: https://drive.google.com/file/d/1od7m1AUA6YinG7qbcXU0pNWWYFSlP37g/view?usp=sharing
   - ASL-DVS: https://drive.google.com/file/d/17E7Doq3F9Cn-QdGlJLmEvHlN5KMs-h9f/view?usp=sharing
   - Event Camera Dataset: https://drive.google.com/file/d/1IlaacDk56pNVHLLHJWFrghZvruaLP-iK/view?usp=sharing
2. Download the datasets and unzip them to the folder (./dataset/)
3. Change the corresponding data path in each .py file



## Train and test

### Training

This is the raw and unintegrated code. If you want to train the model on a dataset, such as N-MNIST, run this code

```shell
>>> cd nMnist
>>> python trainNmnist.py --bs 64 --savepath './ckpt/' --epoch 30 --showFreq 50  --lr 0.1 --cuda '1' --j 4
or just change the file train.sh and run
>>> sh train.sh
```



### Testing

After training, run the following code to generating results. **NOTICE: the default output path is (./dataset/N-MNIST/ResConv/HRPre)**.

```shell
>>> python testNmnist.py
```



### Calculate metrics

After generating results, run calRMSE.py to calculate the metrics. **NOTICE: the output path should be changed**.

```shell
>>> python calRMSE.py
```



### Downstream application

1. Classification

   The classification experiment is done with the N-MNIST, Cifar10-DVS and ASL-DVS datasets. After generating the super-resolution results, the classification can be done by the following code (for the N-MNIST dataset for example).

   ```shell
   >>> cd nMnist
   >>> python trainNmnistClassification.py
   ```

   In the classification experiment, the training sets are the ground truth event streams from the training sets of the super-resolution task. The test sets are the test set of the super-resolution task, containing the LR event streams, HR ground truth event streams, output event streams and others. **The test mode and other parameters is selected in the trainNmnistClassification.py file**.

2. Image reconstruction

   The image reconstruction task is done following the E2VID, the code and the pretrain model is proposed in https://github.com/uzh-rpg/rpg_e2vid.

