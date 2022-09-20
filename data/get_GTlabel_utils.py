import numpy as np
import cv2
import os,sys
sys.path.append('.')
from data.relabel import *

NUM_TRAIN_CLASS = 19
NUM_CLASS = 34


def detect_label_change(img_np,r = 2):
    """
    img_np : image in numpy dtype
    r : select radius,default = 2
    """
    H,W = img_np.shape
    neighbors_num = (2 * r + 1) ** 2
    img_pad_np = np.pad(img_np,((r,r),(r,r)),mode = 'edge')
    img_vols_np = np.zeros((H,W,neighbors_num),dtype = np.uint8)
    multiedge_np = np.zeros((H,W,NUM_TRAIN_CLASS),dtype = np.bool)
    for i in range(0,2 * r + 1):
        for j in range(0 , 2 * r + 1):
            img_vols_np[:,:,i * (2 * r + 1) + j] = img_pad_np[i : i + H ,j : j + W ]
    for k in range (NUM_CLASS):
        name = labels[k].name
        trainId = labels[k].trainId
        ignoreInEval = labels[k].ignoreInEval
        if ignoreInEval == False:
            img_ngb_diff = img_vols_np - img_vols_np[:,:,2 * r * (r + 1) : 2 * r * (r + 1) + 1] # [H,W,C]
            img_ngb_diffsum = (img_ngb_diff != 0).sum(axis = -1)
            # pixel label = k && neighbors exist different label pixels
            edge_H,edge_W = np.where(((img_np == k) & (img_ngb_diffsum > 0)))
            multiedge_np[edge_H,edge_W,trainId] = 1
    return multiedge_np


    







