# FAC-Net
### Official Pytorch Implementation of '[Foreground-Action Consistency Network fo Weakly-supervised Temporal Action Localization]

> **Foreground-Action Consistency Network fo Weakly-supervised Temporal Action Localization**<br>
> Linjiang Huang (CUHK), Hongsheng Li (CUHK)
>
> Paper: 
>

## Prerequisites
### Recommended Environment
* Python 3.6
* Pytorch 1.2
* Tensorboard Logger
* CUDA 10.2

### Data Preparation
1. Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    - We recommend using features and annotations provided by [this repo](https://github.com/sujoyp/wtalc-pytorch).

2. Place the features and annotations inside a new `dataset` folder.

## Usage

### Training
You can easily train the model by running the provided script.

- First, please refer to `train_options.py`. Modify the argument of `dataset-root' to the path of your `dataset` folder.

- Second, modify the argument of `run-type' to 0 (RGB) or 1 (optical flow) and make sure you use different `model-id' for RGB and optical flow.

- Third, runn the command below.

~~~~
$ python train_main.py
~~~~

### Evaulation

- Single stream evaluation

Please first modify the argument of `model-id' to correspond to the training id in `train_main.py', and set `run-type' as 2 (RGB) or 3 (optical flow). 
Run the command below.

~~~~
$ python train_main.py
~~~~

- Two stream evaluation

Please first modify the argument of `rgb-model-id' and `flow-model-id' to correspond to the training ids in `test_main.py', and set `rgb-load-epoch' and `flow-load-epoch' to the epoch of the best model.
Runn the command below.

~~~~
$ python test_main.py
~~~~

## References
We referenced the repos below for the code.

* [STPN](https://github.com/bellos1203/STPN)
* [W-TALC](https://github.com/sujoyp/wtalc-pytorch)
* [BaSNet](https://github.com/Pilhyeon/BaSNet-pytorch)

## Citation
If you find this code useful, please cite our paper.

~~~~
~~~~

## Contact
If you have any question or comment, please contact the first author of the paper - Linjiang Huang (ljhuang@cpii.hk).
