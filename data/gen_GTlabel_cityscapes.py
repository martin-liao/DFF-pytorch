import numpy as np
import cv2
from get_GTlabel_utils import *
import os
import tqdm
from datetime import datetime
from glob import glob
from multiprocessing import Process

def generate_GroundTrueth_Edgemap(city_folder,edge_city_folder,city,r = 2):
    
    semimg_list = glob(os.path.join(city_folder,"*_gtFine_labelIds.png"))
    for j in range(len(semimg_list)):
        name = semimg_list[j].split('/')[-1][: -20]
        time = datetime.now().strftime("%c")
        print("%s | city : %s | [%i/%i] |"%(time,city,j,len(semimg_list)))
        img = cv2.imread(semimg_list[j],cv2.IMREAD_GRAYSCALE)
        multiedge_np = detect_label_change(img,r = r)
        outfile_path = os.path.join(edge_city_folder,"%s_19.npy"%name)
        np.save(outfile_path,multiedge_np)
    


if __name__ == "__main__":
    root_folder = "../cityscapes"
    data_folder = os.path.join(root_folder,"gtFine_trainvaltest","gtFine","train")
    edge_folder = os.path.join(root_folder,"gtFine_trainvaltest","gtFine","edge")
    if os.path.exists(edge_folder) == False:
        os.makedirs(edge_folder)
    citys_list = os.listdir(data_folder)
    r = 2

    cityscapes_threads = []
    for i in range(len(citys_list)):
        city = citys_list[i]
        city_folder = os.path.join(data_folder,city)
        edge_city_folder = os.path.join(edge_folder,city)
        if os.path.exists(edge_city_folder) == False:
            os.makedirs(edge_city_folder)

        cityscapes_threads.append(Process(target = generate_GroundTrueth_Edgemap,
        args = (city_folder,edge_city_folder,city,r)))

for thread in cityscapes_threads:
    thread.start()

for thread in cityscapes_threads:
    thread.join()
        

    

