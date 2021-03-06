# FAC-Net

> **Foreground-Action Consistency Network for Weakly Supervised Temporal Action Localization**<br>
> Linjiang Huang (CUHK), Liang Wang (CASIA), Hongsheng Li (CUHK)
>
> [![arXiv](https://img.shields.io/badge/arXiv-2108.06524-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2108.06524) [![ICCV2021](https://img.shields.io/badge/ICCV-2021-brightgreen.svg?style=plastic)](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Foreground-Action_Consistency_Network_for_Weakly_Supervised_Temporal_Action_Localization_ICCV_2021_paper.pdf)


## Overview
We argue that existing methods for weakly-supervised temporal activity localization cannot guarantee the foreground-action consistency, that is, the foreground and actions are mutually inclusive. Therefore, we propose a novel method named Foreground-Action Consistency Network (FAC-Net) to address this issue. The experimental results on THUMOS14 are as below.

| Method \ mAP(%) | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | AVG |
|:----------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| [UntrimmedNet](https://arxiv.org/abs/1703.03329) | 44.4 | 37.7 | 28.2 | 21.1 | 13.7 | - | - | - |
| [STPN](https://arxiv.org/abs/1712.05080) | 52.0 | 44.7 | 35.5 | 25.8 | 16.9 | 9.9 | 4.3 | 27.0 |
| [W-TALC](https://arxiv.org/abs/1807.10418) | 55.2 | 49.6 | 40.1 | 31.1 | 22.8 | - | 7.6 | - |
| [AutoLoc](https://arxiv.org/abs/1807.08333) | - | - | 35.8 | 29.0 | 21.2 | 13.4 | 5.8 | - | - | - |
| [CleanNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Weakly_Supervised_Temporal_Action_Localization_Through_Contrast_Based_Evaluation_Networks_ICCV_2019_paper.html) | - | - | 37.0 | 30.9 | 23.9 | 13.9 | 7.1 | - |
| [MAAN](https://arxiv.org/abs/1905.08586) | 59.8 | 50.8 | 41.1 | 30.6 | 20.3 | 12.0 | 6.9 | 31.6 |
| [CMCS](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Completeness_Modeling_and_Context_Separation_for_Weakly_Supervised_Temporal_Action_CVPR_2019_paper.pdf) | 57.4 | 50.8 | 41.2 | 32.1 | 23.1 | 15.0 | 7.0 | 32.4 |
| [BM](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_Weakly-Supervised_Action_Localization_With_Background_Modeling_ICCV_2019_paper.pdf) | 60.4 | 56.0 | 46.6 | 37.5 | 26.8 | 17.6 | 9.0 | 36.3 |
| [RPN](https://ojs.aaai.org/index.php/AAAI/article/view/6760/6614) | 62.3 | 57.0 | 48.2 | 37.2 | 27.9 | 16.7 | 8.1 | 36.8 |
| [DGAM](https://dl.acm.org/doi/pdf/10.1145/3343031.3351044) | 60.0 | 54.2 | 46.8 | 38.2 | 28.8 | 19.8 | 11.4 | 37.0 |
| [TSCN](https://arxiv.org/pdf/2010.11594) | 63.4 | 57.6 | 47.8 | 37.7 | 28.7 | 19.4 | 10.2 | 37.8 |
| [EM-MIL](https://arxiv.org/abs/1911.09963) | 59.1 | 52.7 | 45.5 | 36.8 | 30.5 | 22.7 | **16.4** | 37.7 |
| [BaS-Net](https://arxiv.org/abs/1911.09963) | 58.2 | 52.3 | 44.6 | 36.0 | 27.0 | 18.6 | 10.4 | 35.3 |
| [A2CL-PT](https://arxiv.org/pdf/2007.06643) | 61.2 | 56.1 | 48.1 | 39.0 | 30.1 | 19.2 | 10.6 | 37.8 |
| [ACM-BANet](https://dl.acm.org/doi/pdf/10.1145/3394171.3413687) | 64.6 | 57.7 | 48.9 | 40.9 | 32.3 | 21.9 | 13.5 | 39.9 |
| [HAM-Net](https://arxiv.org/pdf/2101.00545) | 65.4 | 59.0 | 50.3 | 41.1 | 31.0 | 20.7 | 11.1 | 39.8 |
| [UM](https://arxiv.org/abs/2006.07006) | 67.5 | 61.2 | 52.3 | 43.4 | **33.7** | **22.9** | 12.1 | 41.9 |
| [**FAC-Net (Ours)**](https://arxiv.org/pdf/2108.06524.pdf) | **67.6** | **62.1** | **52.6** | **44.3** | 33.4 | 22.5 | 12.7 | **42.2** |

## Prerequisites
### Recommended Environment
* Python 3.6
* Pytorch 1.5
* Tensorboard Logger
* CUDA 10.1

### Data Preparation
1. Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    - We recommend using features and annotations provided by [this repo](https://github.com/sujoyp/wtalc-pytorch).

2. Place the [features](https://emailucr-my.sharepoint.com/personal/sujoy_paul_email_ucr_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsujoy%5Fpaul%5Femail%5Fucr%5Fedu%2FDocuments%2Fwtalc%2Dfeatures&originalPath=aHR0cHM6Ly9lbWFpbHVjci1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9zdWpveV9wYXVsX2VtYWlsX3Vjcl9lZHUvRXMxemJIUVk0UHhLaFVrZGd2V0h0VTBCSy1feXVnYVNqWEs4NGtXc0IwWEQwdz9ydGltZT1yUFZFOUUzbDJFZw) and [annotations](https://github.com/sujoyp/wtalc-pytorch/tree/master/Thumos14reduced-Annotations) inside a `dataset/Thumos14reduced/` folder.

## Usage

### Training
You can easily train the model by running the provided script.

- Refer to `train_options.py`. Modify the argument of `dataset-root` to the path of your `dataset` folder.

- Run the command below.

~~~~
$ python train_main.py --run-type 0 --model-id 1   # rgb stream
$ python train_main.py --run-type 1 --model-id 2   # flow stream
~~~~

Make sure you use different `model-id` for RGB and optical flow.
Models are saved in `./ckpt/dataset_name/model_id/`

### Evaulation

#### 
The trained model can be found [here](https://drive.google.com/drive/folders/1XMkyHwtZFJP4CZwK3qBfjrkvJ_RNeUxQ?usp=sharing). Please change the file name to xxx.pkl (e.g., 100.pkl) and put it into `./ckpt/dataset_name/model_id/`. You can evaluate the model referring to the two stream evaluation process.

#### Single stream evaluation

- Run the command below.

~~~~
$ python train_main.py --pretrained --run-type 2 --model-id 1 --load-epoch 100  # rgb stream
$ python train_main.py --pretrained --run-type 3 --model-id 2 --load-epoch 100  # flow stream
~~~~

`load-epoch` refers to the epoch of the best model. The best model would not always occur at 100 epoch, please refer to the log in the same folder of saved models to set the load epoch of the best model.
Make sure you set the right `model-id` that corresponds to the `model-id` during training.


#### Two stream evaluation

- Run the command below using our provided models.

~~~~
$ python test_main.py --rgb-model-id 1 --flow-model-id 2 --rgb-load-epoch 100 --flow-load-epoch 100
~~~~

## References
We referenced the repos below for the code.

* [STPN](https://github.com/bellos1203/STPN)
* [W-TALC](https://github.com/sujoyp/wtalc-pytorch)
* [BaS-Net](https://github.com/Pilhyeon/BaSNet-pytorch)

If you find this code useful, please cite our paper.

~~~~
@InProceedings{Huang_2021_ICCV,
    author    = {Huang, Linjiang and Wang, Liang and Li, Hongsheng},
    title     = {Foreground-Action Consistency Network for Weakly Supervised Temporal Action Localization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {8002-8011}
}
~~~~

## Contact
If you have any question or comment, please contact the first author of the paper - Linjiang Huang (ljhuang524@gmail.com).
