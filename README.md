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
* CUDA 10.0

### Data Preparation
1. Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    - We recommend using features and annotations provided by [this repo](https://github.com/sujoyp/wtalc-pytorch).

2. Place the [features](https://emailucr-my.sharepoint.com/personal/sujoy_paul_email_ucr_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsujoy%5Fpaul%5Femail%5Fucr%5Fedu%2FDocuments%2Fwtalc%2Dfeatures&originalPath=aHR0cHM6Ly9lbWFpbHVjci1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9zdWpveV9wYXVsX2VtYWlsX3Vjcl9lZHUvRXMxemJIUVk0UHhLaFVrZGd2V0h0VTBCSy1feXVnYVNqWEs4NGtXc0IwWEQwdz9ydGltZT1yUFZFOUUzbDJFZw) and [annotations](https://github.com/sujoyp/wtalc-pytorch/tree/master/Thumos14reduced-Annotations) inside a `dataset/Thumos14reduced/` folder.

## Usage

### Training
You can easily train the model by running the provided script.

- Refer to `train_options.py`. Modify the argument of `dataset-root` to the path of your `dataset` folder.

- Modify the argument of `run-type` to 0 (RGB) or 1 (optical flow) and make sure you use different `model-id` for RGB and optical flow.

- Run the command below.

~~~~
$ python train_main.py
~~~~

Models are saved in `./ckpt/dataset_name/model_id/`

### Evaulation

#### 
The pre-trained model can be found [here](https://drive.google.com/drive/folders/1XMkyHwtZFJP4CZwK3qBfjrkvJ_RNeUxQ?usp=sharing). You can evaluate the model refering to the two stream evaluation process.

#### Single stream evaluation

- Modify the argument of `model-id` in `train_options.py` to correspond to the training id
- Set the argument of `load-epoch` to the epoch of the best model
- Set `run-type` as 2 (RGB) or 3 (optical flow). 
- Run the command below.

~~~~
$ python train_main.py
~~~~

#### Two stream evaluation

- Modify the argument of `rgb-model-id` and `flow-model-id` in `test_options.py` to correspond to the training ids

- Set `rgb-load-epoch` to the epoch of the best RGB model and do the same on the `flow-load-epoch`.

- Run the command below.

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
