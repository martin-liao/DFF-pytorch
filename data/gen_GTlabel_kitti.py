import numpy as np
import cv2
from get_GTlabel_utils import *
import os
import tqdm
from datetime import datetime

def generate_GroundTrueth_Edgemap(root_folder,r = 2,training = True):
    if training == True:
        sem_folder = os.path.join(root_folder,"training","semantic")
        edge_folder = os.path.join(root_folder,"training","edge")
    else :
        sem_folder = os.path.join(root_folder,"testing","semantic")
        edge_folder = os.path.join(root_folder,"testing","edge")

    if os.path.exists(edge_folder) == False:
        os.makedirs(edge_folder)
    
    file_num = len(os.listdir(sem_folder))
    for i in range(file_num):
        time = datetime.now().strftime("%c")
        print("%s | [%i/%i] |"%(time,i,file_num))
        file_path = os.path.join(sem_folder,"%06d_10.png"%i)
        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        multiedge_np = detect_label_change(img,r = r)
        outfile_path = os.path.join(edge_folder,"%06d_10.npy"%i)
        np.save(outfile_path,multiedge_np)
    
    print("Done!")


if __name__ == "__main__":
    root_folder = "../kitti-semantic/data_semantics"
    training = True
    r = 2

    generate_GroundTrueth_Edgemap(root_folder = root_folder,r = r,training = training)

    

