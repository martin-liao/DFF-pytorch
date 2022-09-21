# Dynamic Feature Fusion for Semantic Edge Detection (Pure Modern Python,Nonoffical)
Modern  Pytorch implementation of Dynamic Feature Fusion for Semantic Edge detection(IJCAI 2019,nonoffical) and image semantic edge detection part of our novel  calibration work:**Online Calibration of LiDAR and Camera Based on Semantic Edge**(we will released codes after paper accepted).The offical reposity of DFF could be found [here](https://github.com/Lavender105/DFF).
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
This reposity is just a pure Python3 type of DFF without modify.Furthermore,We provide codes for ground truth labeling and prediction images visualization,both KITTI-Odometry Dataset and Cityscapes Dataset,following guide of CASENet.If you want to get into details,please read the original [paper](https://arxiv.org/abs/1705.09759).
## data preprocessing
Please look for data folder and kitti-dataset.py will help you solve it.
## training
We provide training module in root folder,you could directly use it.
## test/predict
We provide test and predict(including visualization part) in root folder,you could directly use it and have fun.
