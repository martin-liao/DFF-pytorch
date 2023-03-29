import numpy as np
from scipy.sparse import coo_matrix
import cv2
from get_GTlabel_utils import *
from data.vis import visulize
import os
import tqdm
from datetime import datetime
from glob import glob
from multiprocessing import Process

def generate_GroundTrueth_Edgemap(city_folder,edge_city_folder,vis_folder,city,r = 2):

    labelimg_list = glob(os.path.join(city_folder,"*_gtFine_labelIds.png"))
    insimg_list = glob(os.path.join(city_folder,"*_gtFine_instanceIds.png"))
    labelimg_list.sort()
    insimg_list.sort()
    for j in range(len(labelimg_list)):
        name = labelimg_list[j].split('/')[-1][: -20]
        time = datetime.now().strftime("%c")
        print("| time : %s | city : %s | [%i/%i] |"%(time,city,j,len(labelimg_list)))
        label_np = cv2.imread(labelimg_list[j],cv2.IMREAD_UNCHANGED)
        ins_np = cv2.imread(insimg_list[j],cv2.IMREAD_UNCHANGED)
        # id --> trainid
        label_np = id2trainid(label_np)
        # generate binary edge map
        edge_binmap = seg2edge(ins_np,r = r,label_ignore = [2,3])
        # initialize multiple semantic edge map
        edge_semmap = np.zeros((NUM_TRAIN_CLASS,label_np.shape[0],label_np.shape[1]),dtype = np.bool_)
        edge_semmap_splist = []

        for idx_cls in range(NUM_TRAIN_CLASS):
            idx_seg = label_np == idx_cls
            if(idx_seg.sum() != 0):
                seg_map = np.zeros_like(label_np)
                seg_map[idx_seg] = ins_np[idx_seg]
                edge_semmap[idx_cls] = seg2edge_fast(seg_map,edge_binmap,2,[])
                edge_semmap_splist.append(coo_matrix(edge_semmap[idx_cls]))
            else:
                edge_semmap_splist.append(coo_matrix(np.zeros_like(label_np,dtype = np.bool_)))    
        edge_semmap_sparr = np.array(edge_semmap_splist)

        # visualize generated multi-semantic edge map
        edge_colormap = visulize(edge_semmap)
        color_savepath = os.path.join(vis_folder,"%s_19edgecolor.png"%name)
        cv2.imwrite(color_savepath,edge_colormap)

        # change to sparse matrix for storeage effiency
        # edge_semmap = coo_matrix(edge_semmap)
        edge_savepath = os.path.join(edge_city_folder,"%s_19"%name)
        np.savez_compressed(edge_savepath,edge_semmap_sparr)


root_folder = "/home/martin/Documents/data/image_edge_dataset/cityscapes/data_orig"
data_folder = os.path.join(root_folder,"gtFine","train")
edge_folder = os.path.join(root_folder,"gtFine","train_edge")
vis_folder = os.path.join(root_folder,"gtFine","train_edgevis")
if os.path.exists(edge_folder) == False:
    os.makedirs(edge_folder)
if os.path.exists(vis_folder) == False:
    os.makedirs(vis_folder)
citys_list = os.listdir(data_folder)
citys_list.sort()
r = 2

cityscapes_threads = []
for i in range(len(citys_list)):
    city = citys_list[i]
    city_folder = os.path.join(data_folder,city)
    edge_city_folder = os.path.join(edge_folder,city)
    vis_city_folder = os.path.join(vis_folder,city)
    if os.path.exists(edge_city_folder) == False:
        os.makedirs(edge_city_folder)
    if os.path.exists(vis_city_folder) == False:
        os.makedirs(vis_city_folder)

    cityscapes_threads.append(Process(target = generate_GroundTrueth_Edgemap,
    args = (city_folder,edge_city_folder,vis_city_folder,city,r)))

for thread in cityscapes_threads:
    thread.start()

for thread in cityscapes_threads:
    thread.join()
        

    

