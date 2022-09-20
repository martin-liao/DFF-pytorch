from re import I
from torch.utils.data import DataLoader
import wandb
from loss.loss import reweighted_Cross_Entroy_Loss
from models.dff import DFF
from dataset.semantic_dataset import kitti_Odometry_Dataset
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
    preds = np.where(preds > 0.45,1,0).astype(np.uint8)
    for i in range(NUM_class):
        color = mini_label[i][2]
        edge = preds[i,:,:]
        edge_sum += edge
        masked_image = apply_mask(image, edge, color)

    edge_sum = np.array([edge_sum, edge_sum, edge_sum])
    edge_sum = np.transpose(edge_sum, (1, 2, 0))
    idx = edge_sum > 0
    masked_image[idx] = masked_image[idx]/edge_sum[idx]
    masked_image[~idx] = 255
    masked_image = masked_image.astype(np.uint8)

    io.imsave(out_path, masked_image)
        
        


class Predictor():
    def __init__(self,
    root_path : str = "../KITTI-odometry/data_odometry_color/sequences",
    pretrained_path : str = "./checkpoint/dff_cityscapes_resnet101.pth.tar",
    size : tuple = (320,1024),
    seq_list = list(range(7,8)),
    batch_size : int = 4,
    shuffle : bool = False,
    num_workers : int = 2,
    device  : str = "cuda"):

        # load dataset
        self.root_path = root_path
        self.seq_path_list = []
        self.seq_vis_path_list = []
        # make edge folders
        for seq in seq_list:
            seq_folder = os.path.join(self.root_path,"%02d"%seq,"img2_edge")
            seq_vis_folder = os.path.join(self.root_path,"%02d"%seq,"img2_edgevis")

            if os.path.exists(seq_folder) == False:
                os.makedirs(seq_folder)
            self.seq_path_list.append(seq_folder)

            if os.path.exists(seq_vis_folder) == False:
                os.makedirs(seq_vis_folder)
            self.seq_vis_path_list.append(seq_vis_folder)
        
        self.dataset = kitti_Odometry_Dataset(root_path,size = size,seq_list = seq_list)
        length = len(self.dataset.dataset)

        self.dataloader = DataLoader(self.dataset,
        batch_size = batch_size,shuffle = shuffle,num_workers = num_workers)

        # load model
        self.model = DFF(nclass = NUM_class,pretrained = True,device = device)

        # load pretrained model
        if pretrained_path is not None:
            pretrained_model = torch.load(pretrained_path)['state_dict']
            self.model.load_state_dict(pretrained_model)
        
        self.device = device
    
    def pred(self):

        self.model.eval()
        with torch.no_grad():
            for i ,(seq,idx,img) in enumerate(self.dataloader):
                img = img.to(self.device)
                side5s,fuses = self.model(img)
                for j in range(fuses.shape[0]):
                    fuse_np = fuses[j].sigmoid_().cpu().numpy() # [C,H,W]
                    
                    fuse_path = os.path.join(self.seq_path_list[0],"%06i_edge.npy"%idx[j].item())
                    fuse_vispath = os.path.join(self.seq_vis_path_list[0],"%06i_edgevis.png"%idx[j].item())
                    np.save(fuse_path,fuse_np)
                    vis_pred(fuse_np,fuse_vispath)

                    now = datetime.now()
                    log = "test: {} | Batch = [{:03d}/{:03d}] | seq = {:02d} | idx = {:06d} |"
                    log = log.format(now.strftime("%c"),i,len(self.dataloader),seq[j].item(),idx[j].item())
                    print(log)

if __name__ == "__main__":
    preder = Predictor()
    preder.pred()
                







