# Dynamic Feature Fusion for Semantic Edge Detection (Pure Modern Python)
This repository is a modern  Pytorch implementation of Dynamic Feature Fusion for Semantic Edge detection (DFF,IJCAI 2019), and also image semantic edge detection part of our novel multi-modalities calibration work (we will released codes after the paper accepted).The offical reposity of DFF could be found [here](https://github.com/Lavender105/DFF). We remove the pytorch_encoding module in the official version but keep the network exactly the same as raw. You could utilize the pretrained model in officil repository in our version directly and have fun !
![flow](https://github.com/Lavender105/DFF/blob/master/img/overview.png?raw=true)
![arc](https://github.com/Lavender105/DFF/raw/master/img/visualization.png)
## Usage
### Prerequisites
- Pytorch(>=1.4.0)
- Python-Opencv
- Open3D
- wandb(optional)
- scipy
- skimage
### Pre-trained models
The official project released their pretrained model on SBD and Cityscapes dataset.You could find them [here](https://drive.google.com/open?id=1-PCfJH6w1sFE5Q-B-GXL_D9DiiEyWA7P) and [here](https://drive.google.com/open?id=1-PCfJH6w1sFE5Q-B-GXL_D9DiiEyWA7P)(code: n24x).
### !!Warning
If you want to predict semantic edge on KITTI dataset,we suggest to transform the pretrained model of Cityscapes directly rather than training on KITTI-Semantic Dataset.We have test several times ,including training from scratch,fine-tuning based on pretrained model,all of them failed without exception.The reason is resolution of images provided by KITTI Dataset is horrible(only 370*1220,approxiamately 1/4 of Cityscapes). 
### Notation
This reposity is just a pure Python3 type of DFF without modify.Furthermore,We provide Python3 codes for ground truth labeling and prediction images visualization,both KITTI-Odometry Dataset and Cityscapes Dataset,following the guidance of CASENet. Please read the [paper] to get into details (https://arxiv.org/abs/1705.09759).
## data preprocessing
Please look for data folder and find ..._GTlabel....py. This script is just a python3 implementation of GT generation files in DFF, but slightly different from those in CaseNet (instance-aware label vs instance-unaware label). 
## training
We provide training module in root folder,you could directly use it. We will provide a more effective pytorch-lightning trainer module in the future.
## test/predict
We provide test and predict(including visualization part) in root folder,you could directly use it and have fun.
