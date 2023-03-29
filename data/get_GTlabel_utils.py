import numpy as np
import cv2
import os,sys
sys.path.append('.')
from data.relabel import *

NUM_TRAIN_CLASS = 19
NUM_CLASS = 34

def id2trainid(id_np):
    trainId_map = np.ones_like(id_np) * 255
    
    for k in range(NUM_CLASS):
        id = labels[k].id
        trainId = labels[k].trainId
        ignoreInEval = labels[k].ignoreInEval
        if ignoreInEval == False:
            trainId_map[id_np == id] = trainId
    return trainId_map

def seg2edge(ins_np,r = 2,label_ignore = None):
    """
    ins_np : instance map, H * W
    r : search radius,default = 2
    label_ignore: ignore label map
    """
    H,W = ins_np.shape
    r = max(r,1)
    X,Y = np.meshgrid(np.arange(0,W),np.arange(0,H))
    x,y = np.meshgrid(np.arange(-r,r + 1),np.arange(-r,r + 1))

    # vectorize everything
    X,Y = X.flatten(),Y.flatten() # [H * W,1]
    x,y = x.flatten(),y.flatten() # [(2r + 1) ^ 2,1]
    ins_np = ins_np.flatten()

    valid = x ** 2 + y ** 2 <= r ** 2
    x = x[valid]
    y = y[valid]
    numPxlImage = len(X)
    numPxlNeigh = len(x)

    idxEdge = np.zeros((numPxlImage,),dtype = np.int16)
    for i in range(numPxlNeigh):
        XNeigh = X+x[i]
        YNeigh = Y+y[i]
        idxValid = np.where(np.logical_and((XNeigh >= 0) & (XNeigh < W),(YNeigh >= 0) & (YNeigh < H)))[0]
    
        XCenter = X[idxValid]
        YCenter = Y[idxValid]
        XNeigh = XNeigh[idxValid]
        YNeigh = YNeigh[idxValid]
        LCenter = ins_np[np.ravel_multi_index([YCenter,XCenter],(H,W))]
        LNeigh = ins_np[np.ravel_multi_index([YNeigh,XNeigh],(H,W))]

        idx_Diff = np.where(LCenter != LNeigh)[0]

        if idx_Diff.sum() == 0:
            continue
        # remove edge pixels with ignore label in center or neighborhood
        LCenterEdge = LCenter[idx_Diff]
        LNeighEdge = LNeigh[idx_Diff]
        idxIgnore = np.zeros_like(LCenterEdge,dtype = np.bool_)
        for j in range(len(label_ignore)):
            idxIgnore += np.logical_or((LCenterEdge == label_ignore[j]),(LNeighEdge == label_ignore[j]))
        idx_DiffGT = idx_Diff[~idxIgnore]
        idxEdge[idxValid[idx_DiffGT]] = True
        
    idxEdge = idxEdge.reshape(H,W)
    return idxEdge

def seg2edge_fast(seg_np,edge_bin,r = 2,label_ignore = None):
    """
    convert seg map to semantic edge map fastly (based on binary edge map)
    """
    H,W = seg_np.shape
    r = max(r,1)
    X,Y = np.meshgrid(np.arange(0,W),np.arange(0,H))
    x,y = np.meshgrid(np.arange(-r,r + 1),np.arange(-r,r + 1))

    # vectorize everything
    X,Y = X.flatten(),Y.flatten() # [H * W,1]
    x,y = x.flatten(),y.flatten() # [(2r + 1) ^ 2,1]
    seg_np = seg_np.flatten()
    edge_bin = edge_bin.flatten() != 0
    X_cand = X[edge_bin]
    Y_cand = Y[edge_bin]
    idx_cand = np.where(edge_bin)[0]
    

    valid = x ** 2 + y ** 2 <= r ** 2
    x = x[valid]
    y = y[valid]
    numPxlImage = len(X)
    numPxlNeigh = len(x)

    idxEdge = np.zeros((numPxlImage,1))
    for i in range(numPxlNeigh):
        XNeigh = X_cand+x[i]
        YNeigh = Y_cand+y[i]
        idxSelect = np.logical_and((XNeigh >= 0) & (XNeigh < W),(YNeigh >= 0) & (YNeigh < H))
        idxValid = idx_cand[idxSelect]

        XCenter = X[idxValid]
        YCenter = Y[idxValid]
        XNeigh = XNeigh[idxSelect]
        YNeigh = YNeigh[idxSelect]
        LCenter = seg_np[np.ravel_multi_index([YCenter,XCenter],(H,W))]
        LNeigh = seg_np[np.ravel_multi_index([YNeigh,XNeigh],(H,W))]

        idx_Diff = np.where(LCenter != LNeigh)[0]
        if idx_Diff.sum() == 0:
            continue
        # remove edge pixels with ignore label in center or neighborhood
        LCenterEdge = LCenter[idx_Diff]
        LNeighEdge = LNeigh[idx_Diff]
        idxIgnore = np.zeros_like(LCenterEdge,dtype = np.bool_)
        for j in range(len(label_ignore)):
            idxIgnore += np.logical_or(LCenterEdge == label_ignore[j],LNeighEdge == idxIgnore[j])
        idx_DiffGT = idx_Diff[~idxIgnore]
        idxEdge[idxValid[idx_DiffGT]] = True
    
    idxEdge = idxEdge.reshape(H,W)
    
    return idxEdge


def detect_label_change(seg_np,r = 2):
    """
    img_np : image in numpy dtype
    r : select radius,default = 2
    """
    H,W = seg_np.shape
    neighbors_num = (2 * r + 1) ** 2
    x,y = np.meshgrid(np.linspace(-r,r),np.linspace(-r,r))
    # select x and y in neighborhood circle
    valid = np.where(x ** 2 + y ** 2 <= r ** 2)
    valid_x = x[valid]
    valid_y = y[valid]
    # pad the origin seg map for neighborhood pixels selection
    img_pad_np = np.pad(seg_np,((r,r),(r,r)),mode = 'edge') 
    img_vols_np = np.zeros((H,W,len(valid)),dtype = np.uint8)
    multiedge_np = np.zeros((H,W,NUM_TRAIN_CLASS),dtype = np.bool_)
    for i in range(0,len(valid)):
        img_vols_np[:,:,i] = img_pad_np[valid_x[i] : valid_x[i] + H ,valid_y[i] : valid_y[i] + W]
    for k in range (NUM_CLASS):
        name = labels[k].name
        trainId = labels[k].trainId
        ignoreInEval = labels[k].ignoreInEval
        if ignoreInEval == False:
            img_ngb_diff = img_vols_np - img_vols_np[:,:,2 * r * (r + 1) : 2 * r * (r + 1) + 1] # [H,W,C]
            img_ngb_diffsum = (img_ngb_diff != 0).sum(axis = -1)
            # pixel label = k && neighbors exist different label pixels
            edge_H,edge_W = np.where(((seg_np == k) & (img_ngb_diffsum > 0)))
            multiedge_np[edge_H,edge_W,trainId] = 1
    return multiedge_np