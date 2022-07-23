# RHDN
The python code implementation of the paper "A Spatio-Spectral Fusion Method for Hyperspectral Images Using Residual Hyper-Dense Network"

# Requirements

- Ubuntu 20.04   cuda 11.0
- Python 3.8  Pytorch 1.7

# Usage

## Brief description

- __data__ floder stores training and testing dataset.
- __fusion__ floder stores the fused data of network test.
- __weights__ floder stores optimal network training parameters.
- __Attention.py__ provides SpatialAttention and ChannelAttention modules.
- __dataloader.py__ generates data iterator.
- __Model.py__ defines the Residual Hyper-Dense Network(RHDN).
- __Model_train.py__ uses __Train__ and __Test__ flags to control model training and testing.
- More details are commented in the code.

## Sample Test

1. The __Test__ and __Train__ flags set to __True__ and __False__ in __Model_train.py__.
2. Run __Model_train.py__ to load the __net_weihts.pth__ to obtain the fused data.

**explain**

- Due to the limitation of github upload capacity, we only upload five sample images of Pavia.
- Note that you can download all the test and fused images of Pavia from Baidu Cloud links:[https://pan.baidu.com/s/1ytquzgD_Jvwa2czJPjElXQ](https://pan.baidu.com/s/1ytquzgD_Jvwa2czJPjElXQ)(Access Code:wyw2)

# Citation
If you find this code helpful, please kindly cite:

@article{qu2021,

title={A Spatio-Spectral Fusion Method for Hyperspectral Images Using Residual Hyper-Dense Network},

author={Jiahui Qu, Yanzi Shi, Weiying Xie, Yunsong Li, Xianyun Wu, and Qian Du},

journal={IEEE Transactions on Neural Networks and Learning Systems},

year={2022}}




