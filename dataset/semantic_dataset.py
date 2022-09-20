import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random
import torch
from torchvision.transforms import transforms
from scipy.ndimage import zoom

NUM_CLASS = 19

CITYS_LIST = [
    "aachen",
    "bochum",
    "bremen",
    "cologne",
    "darmstadt",
    "dusseldorf",
    "erfurt",
    "hamburg",
    "hanover",
    "jena",
    "krefeld",
    "monchengladbach",
    "strasbourg",
    "stuttgart",
    "tubingen",
    "ulm",
    "weimar",
    "zurich",

]

def make_kitti_dataset(root_folder):
    dataset = []
    data_folder = os.path.join(root_folder,"training")
    img_folder = os.path.join(data_folder,"image_2")
    edge_folder = os.path.join(data_folder,"edge")
    filenum = len(os.listdir(img_folder))

    for i in range(filenum):
        img_path = os.path.join(img_folder,"%06d_10.png"%i)
        sem_path = os.path.join(edge_folder,"%06d_10.npy"%i)
        dataset.append((i,img_path,sem_path))

    return dataset
    
def make_cityscapes_dataset(root_folder):
    dataset = []
    data_folder = os.path.join(root_folder,"leftImg8bit_trainvaltest","leftImg8bit","train")
    edge_folder = os.path.join(root_folder,"gtFine_trainvaltest","gtFine","edge")
    for city in CITYS_LIST:
        data_city_folder = os.path.join(data_folder,city)
        edge_city_folder = os.path.join(edge_folder,city)
        # img_city_list = glob(data_city_folder + "*.png")
        edge_city_list = os.listdir(edge_city_folder)
        for i in range(len(edge_city_list)):
            edge_city_path = edge_city_list[i]
            edge_city_fullpath = os.path.join(edge_city_folder,edge_city_path)
            img_city_path = os.path.join(data_city_folder,edge_city_path.replace
            ("19.npy","leftImg8bit.png"))
            dataset.append((i,img_city_path,edge_city_fullpath))
    
    return dataset

def make_kitti_odometry_dataset(root_folder,seq_list):
    dataset = []
    for seq in seq_list:
        seq_img2_folder = os.path.join(root_folder,"%02d"%seq,"image_2")
        img_num = len(os.listdir(seq_img2_folder))
        for i in range(img_num):
            img_path = os.path.join(seq_img2_folder,"%06d.png"%i)
            dataset.append((seq,i,img_path))
    
    return dataset





class Dataset(Dataset):
    def __init__(self,root_folder,scale = [0.8,1.0,1.25],size = (320,1024),dataset = "kitti",mode = "train"):
        self.scale = scale
        self.size = size
        self.mode = mode
        if dataset == "kitti":
            self.dataset = make_kitti_dataset(root_folder = root_folder)
        
        elif dataset == "cityscapes":
            self.dataset = make_cityscapes_dataset(root_folder = root_folder)
        

        self.transform = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        i,img_path,target_path = self.dataset[index]

        img_np = cv2.imread(img_path)
        img_np = img_np[:,:,::-1]
        # img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB) # BGR->RGB
        tar_np = np.load(target_path)

        if self.mode == "train":
            # mirror
            """
            mir = random.randint(0,1)
            if mir == 1:
                img_np = cv2.flip(img_np,1)
                tar_np = cv2.flip(tar_np,1)

            # resize
            scale = random.choice(self.scale)
            reshape = (int(round(tar_np.shape[1] * scale)),
                    int(round(tar_np.shape[0] * scale)))

            img_np = cv2.resize(
                img_np,
                reshape,
                interpolation = cv2.INTER_LINEAR
            )

            tar_np = zoom(tar_np,
            [scale,scale,1.0],
            order = 0
            )

            """
    
            # random crop
            beg_u = random.randint(0,img_np.shape[1] - self.size[1])
            beg_v = random.randint(0,img_np.shape[0] - self.size[0])

            img_np = img_np[beg_v : beg_v + self.size[0],
                        beg_u : beg_u + self.size[1],
                        :]

            tar_np = tar_np[beg_v : beg_v + self.size[0],
                        beg_u : beg_u + self.size[1],
                        :]

        
        elif self.mode == "test":
            # resize
            scale = (float(self.size[0] / img_np.shape[0]),
                     float(self.size[1] / img_np.shape[1]))

            img_np = cv2.resize(
                img_np,
                (self.size[1],self.size[0]),
                interpolation = cv2.INTER_LINEAR
            )

            tar_np = zoom(tar_np,
            [scale[0],scale[1],1.0],
            order = 0
            )

        img = self.transform(img_np)

        tar = torch.from_numpy(tar_np)
        
        tar = tar.permute(2,0,1) # channel first

        
        return i,img,tar
        

class kitti_Odometry_Dataset(nn.Module):
    def __init__(self,root_folder,size = (320,1024),seq_list = list(range(11))):
        self.dataset = make_kitti_odometry_dataset(root_folder = root_folder,seq_list = seq_list)
        self.size = size
        self.transform = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        seq,i,img2_path = self.dataset[idx]
        img2_np = cv2.imread(img2_path)
        img2_np = img2_np[:,:,::-1].copy()

        # crop
        H,W,C = img2_np.shape
        h_low = H - self.size[0]
        w_low = (W - self.size[1]) // 2
        
        img2_np = img2_np[h_low : h_low + self.size[0],w_low : w_low + self.size[1],:]
        img2 = self.transform(img2_np)

        return seq,i,img2
