# DFF-pytorch
Modern  Pytorch implementation of Dynamic Feature Fusion for Semantic Edge detection(IJCAI 2019,nonoffical) and image semantic edge detection part of our new work:**Online Calibration of LiDAR and Camera Based on Semantic Edge**(we will released this work after papered accepted).The offical reposity could be found [here](https://github.com/Lavender105/DFF).
## Usage
### Prerequisites
- Pytorch(>=1.4.0)
- Python-Opencv
- Open3d
- wandb(optional)
- scipy
- skimage
### Pre-trained models
The official project released their pretrained model on SBD and Cityscapes dataset.You could find them [here](https://drive.google.com/open?id=1-PCfJH6w1sFE5Q-B-GXL_D9DiiEyWA7P) and [here](https://drive.google.com/open?id=1-PCfJH6w1sFE5Q-B-GXL_D9DiiEyWA7P)(code: n24x).
### Notation
This reposity is just a pure Python3 type of DFF but not a modified one .Furthermore,We provide codes for ground truth labeling and visualization,following  guide of CASENet.If you want to get the details,please read the original [paper](https://arxiv.org/abs/1705.09759).
