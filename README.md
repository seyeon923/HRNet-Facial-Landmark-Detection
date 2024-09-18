# High-resolution networks (HRNets) for facial landmark detection

## Introduction

This is an fork of the official code of [High-Resolution Representations for Facial Landmark Detection](https://arxiv.org/pdf/1904.04514.pdf).
The original code can be checked from <https://github.com/HRNet/HRNet-Facial-Landmark-Detection>.

## Performance

### ImageNet pretrained models

HRNetV2 ImageNet pretrained models are now available! Codes and pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification)

We adopt **HRNetV2-W18**(#Params=9.3M, GFLOPs=4.3G) for facial landmark detection on COFW, AFLW, WFLW and 300W.

### COFW

The model is trained on COFW _train_ and evaluated on COFW _test_.

|    Model    | NME  | FR<sub>0.1</sub> |                        pretrained model                        |                              model                               |
| :---------: | :--: | :--------------: | :------------------------------------------------------------: | :--------------------------------------------------------------: |
| HRNetV2-W18 | 3.45 |       0.20       | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [HR18-COFW.pth](https://1drv.ms/u/s!AiWjZ1LamlxzdFIsEUQl8jgUaMk) |

### AFLW

The model is trained on AFLW _train_ and evaluated on AFLW _full_ and _frontal_.

|    Model    | NME<sub>_full_</sub> | NME<sub>_frontal_</sub> |                        pretrained model                        |                              model                               |
| :---------: | :------------------: | :---------------------: | :------------------------------------------------------------: | :--------------------------------------------------------------: |
| HRNetV2-W18 |         1.57         |          1.46           | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [HR18-AFLW.pth](https://1drv.ms/u/s!AiWjZ1Lamlxzc7xumEw810iBLTc) |

### WFLW

|     NME     | _test_ | _pose_ | _illumination_ | _occlution_ | _blur_ | _makeup_ | _expression_ |                        pretrained model                        |                              model                               |
| :---------: | :----: | :----: | :------------: | :---------: | :----: | :------: | :----------: | :------------------------------------------------------------: | :--------------------------------------------------------------: |
| HRNetV2-W18 |  4.60  |  7.86  |      4.57      |    5.42     |  5.36  |   4.26   |     4.78     | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [HR18-WFLW.pth](https://1drv.ms/u/s!AiWjZ1LamlxzdTsr_9QZCwJsn5U) |

### 300W

|     NME     | _common_ | _challenge_ | _full_ | _test_ |                        pretrained model                        |                              model                               |
| :---------: | :------: | :---------: | :----: | :----: | :------------------------------------------------------------: | :--------------------------------------------------------------: |
| HRNetV2-W18 |   2.91   |    5.11     |  3.34  |  3.85  | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [HR18-300W.pth](https://1drv.ms/u/s!AiWjZ1LamlxzeYLmza1XU-4WhnQ) |

![](images/face.png)

## Quick start

### Install

1. Install PyTorch following the [official instructions](https://pytorch.org/)
2. Install the package

```sh
pip install -e .
```

### HRNetV2 pretrained models

```sh
cd HRNet-Facial-Landmark-Detection
# Download pretrained models into this folder
mkdir hrnetv2_pretrained
```

### Data

1. You need to download the annotations files which have been processed from [OneDrive](https://1drv.ms/u/s!AiWjZ1LamlxzdmYbSkHpPYhI8Ms), [Cloudstor](https://cloudstor.aarnet.edu.au/plus/s/m9lHU2aJId8Sh8l), and [BaiduYun(Acess Code:ypxg)](https://pan.baidu.com/s/1Yg1IEp3l2IpGPolpUsWdfg).

2. You need to download images (300W, AFLW, WFLW) from official websites and then put them into `images` folder for each dataset.

Your `data` directory should look like this:

```
HRNet-Facial-Landmark-Detection
-- src
-- experiments
-- tools
-- data
   |-- 300w
   |   |-- face_landmarks_300w_test.csv
   |   |-- face_landmarks_300w_train.csv
   |   |-- face_landmarks_300w_valid.csv
   |   |-- face_landmarks_300w_valid_challenge.csv
   |   |-- face_landmarks_300w_valid_common.csv
   |   |-- images
   |-- aflw
   |   |-- face_landmarks_aflw_test.csv
   |   |-- face_landmarks_aflw_test_frontal.csv
   |   |-- face_landmarks_aflw_train.csv
   |   |-- images
   |-- cofw
   |   |-- COFW_test_color.mat
   |   |-- COFW_train_color.mat
   |-- wflw
   |   |-- face_landmarks_wflw_test.csv
   |   |-- face_landmarks_wflw_test_blur.csv
   |   |-- face_landmarks_wflw_test_expression.csv
   |   |-- face_landmarks_wflw_test_illumination.csv
   |   |-- face_landmarks_wflw_test_largepose.csv
   |   |-- face_landmarks_wflw_test_makeup.csv
   |   |-- face_landmarks_wflw_test_occlusion.csv
   |   |-- face_landmarks_wflw_train.csv
   |   |-- images

```

### Train

Please specify the configuration file in `experiments` (learning rate should be adjusted when the number of GPUs is changed).

```sh
python tools/train.py --cfg <CONFIG-FILE>
# example:
python tools/train.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml
```

### Test

```sh
python tools/test.py --cfg <CONFIG-FILE> --model-file <MODEL WEIGHT>
# example:
python tools/test.py --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml --model-file HR18-WFLW.pth
```
