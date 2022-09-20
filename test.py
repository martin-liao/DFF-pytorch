from torch.utils.data import DataLoader
import wandb
from loss.loss import reweighted_Cross_Entroy_Loss
from models.dff import DFF
from dataset.semantic_dataset import Dataset
from datetime import datetime
import os
import torch
import numpy as np
from data.vis import mini_label
from skimage import io

NUM_class = 19

def apply_mask(image, mask, color):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[: ,: ,c] + color[c],
                                  image[:, :, c])
    return image

def vis_pred(preds,out_path):
    """
    preds : C * H * W
    """
    C,H,W = preds.shape
    edge_sum = np.zeros((H,W),dtype = np.uint8)
    image = np.zeros((H,W,3),dtype = np.float32)
    # preds = np.where(preds > 0.5,1,0).astype(np.uint8)
    for i in range(NUM_class):
        color = mini_label[i][2]
        edge = preds[i,:,:].astype(np.uint8)
        edge_sum += edge
        masked_image = apply_mask(image, edge, color)

    edge_sum = np.array([edge_sum, edge_sum, edge_sum])
    edge_sum = np.transpose(edge_sum, (1, 2, 0))
    idx = edge_sum > 0
    masked_image[idx] = masked_image[idx]/edge_sum[idx]
    masked_image[~idx] = 255
    masked_image = masked_image.astype(np.uint8)

    if out_path is not None:
        io.imsave(out_path, masked_image)

    return masked_image
        
        


class Tester():
    def __init__(self,
    root_path : str = "../cityscapes",
    pretrained_path : str = "./checkpoint/dff_cityscapes_resnet101.pth.tar",
    dataset : str = "cityscapes",
    mode : str = "test",
    scale : list = [1.0],
    size : tuple = (1024,2048),
    batch_size : int = 1,
    shuffle : bool = False,
    num_workers : int = 4,
    device  : str = "cpu"):

        # load dataset
        self.root_path = root_path
        self.dataset = Dataset(root_path,scale = scale,size = size,dataset = dataset,mode = mode)
        length = len(self.dataset.dataset)

        self.dataloader = DataLoader(self.dataset,
        batch_size = batch_size,shuffle = shuffle,num_workers = num_workers)

        # load model
        self.model = DFF(nclass = NUM_class,pretrained = True,device = device)

        # load pretrained model
        if pretrained_path is not None:
            pretrained_model = torch.load(pretrained_path)['state_dict']
            self.model.load_state_dict(pretrained_model,strict = True)
        
        self.device = device
    
    def test(self):
        fuse_folder = os.path.join(self.root_path,"gtFine_trainvaltest","gtFine","edge_fusepred")
        side5_folder = os.path.join(self.root_path,"gtFine_trainvaltest","gtFine","edge_side5pred")
        gt_folder = os.path.join(self.root_path,"gtFine_trainvaltest","gtFine","edge_gt")
        if os.path.exists(fuse_folder) == False:
            os.makedirs(fuse_folder)
        if os.path.exists(side5_folder) == False:
            os.makedirs(side5_folder)
        if os.path.exists(gt_folder) == False:
            os.makedirs(gt_folder) 


        self.model.eval()
        with torch.no_grad():
            for i ,(idxs,imgs,tars) in enumerate(self.dataloader):
                imgs = imgs.to(self.device)
                tars = tars.to(self.device)
                side5s,fuses = self.model(imgs)
                for j in range(fuses.shape[0]):
                    gt_path = os.path.join(gt_folder,"%06i_10.png"%idxs[j])
                    fuse_path = os.path.join(fuse_folder,"%06i_10.png"%idxs[j])
                    side5_path = os.path.join(side5_folder,"%06i_10.png"%idxs[j])

                    gt_np = tars[j].cpu().numpy()
                    fuse_np = fuses[j].sigmoid_().cpu().numpy() # [C,H,W]
                    side5_np = side5s[j].sigmoid_().cpu().numpy()
                    fuse_np = np.where(fuse_np > 0.45,1.0,0.0)
                    side5_np = np.where(side5_np > 0.45,1.0,0.0)
                
                    vis_pred(gt_np,gt_path)
                    vis_pred(fuse_np,fuse_path)
                    vis_pred(side5_np,side5_path)

                    now = datetime.now()
                    log = "test: {} | Batch = [{:03d}/{:03d}] |"
                    log = log.format(now.strftime("%c"),i,len(self.dataloader))
                    print(log)

if __name__ == "__main__":
    pretrained_path = "./checkpoint/dff_cityscapes_resnet101.pth.tar"
    tester = Tester(pretrained_path = pretrained_path)
    tester.test()
                







