import numpy as np
import cv2
import os
from datetime import datetime
from multiprocessing import Process

NUM_CLASS = 19

mini_label = [
    (0,'road',(128,64,128)),
    (1,'sidewalk',(244,35,232)),
    (2,'building',( 70, 70, 70)),
    (3,'wall',(102,102,156)),
    (4,'fence',(190,153,153)),
    (5,'pole',(153,153,153)),
    (6,'traffic light',(250,170, 30)),
    (7,'traffic sign',(220,220,  0)),
    (8,'vegetation',(107,142, 35)),
    (9,'terrain',(152,251,152)),
    (10,'sky',( 70,130,180)),
    (11,'person',(220, 20, 60)),
    (12,'rider',(255,  0,  0)),
    (13,'car', (  0,  0,142)),
    (14,'truck',(  0,  0, 70)),
    (15,'bus',(  0, 60,100)),
    (16,'train', (  0, 80,100)),
    (17,'motorcycle',(  0,  0,230)),
    (18,'bicycle',(119, 11, 32))
]

def visulize(emulmap_np):
    H,W,C = emulmap_np.shape
    edge_color = np.ones((H,W,3),dtype = np.uint8) * 255
    for i in range(NUM_CLASS):
        emap_np = emulmap_np[:,:,i]
        color = np.array(mini_label[i][2],dtype = np.uint8)
        edge_H,edge_W = np.where(emap_np == 1)
        edge_color[edge_H,edge_W,:] = color
    return edge_color


def vis_kitti(root_path):
    emap_folder = os.path.join(root_path,"edge")
    file_num = len(os.listdir(emap_folder))
    out_folder = os.path.join(root_path,"edge_rgb")
    if os.path.exists(out_folder) == False:
        os.makedirs(out_folder)

    for i in range(file_num):
        log = "%s [%02i/%i]"
        log = log.format(datetime.now().strftime("%c"),i,file_num)
        print(log) 
        emulmap_path = os.path.join(emap_folder,"%06i_10.npy"%i)
        out_path = os.path.join(out_folder,"%06i_10.png"%i)
        emulmap_np = np.load(emulmap_path)
        edge_color_np = visulize(emulmap_np)
        cv2.imwrite(out_path,edge_color_np)

def vis_cityscapes(root_path):
    emap_folder = os.path.join(root_path,"edge")
    citys_list = os.listdir(emap_folder)
    citys_num = len(citys_list)

    out_folder = os.path.join(root_path,"edge_rgb")
    if os.path.exists(out_folder) == False:
        os.makedirs(out_folder)
    
    citys_threads = []
    for i in range(citys_num):
        city = citys_list[i]
        city_folder = os.path.join(emap_folder,city)
        out_city_folder = os.path.join(out_folder,city)
        if os.path.exists(out_city_folder) == False:
            os.makedirs(out_city_folder)
        
        citys_threads.append(Process(target = vis_city,
        args = (city_folder,out_city_folder,city)))
    
    for thread in citys_threads:
        thread.start()
    
    for thread in citys_threads:
        thread.join()

def vis_city(city_folder,out_folder,city):
    file_list = os.listdir(city_folder)
    for i in range(len(file_list)):
        log = "%s | city = %s | [%02i / %02i]"%(datetime.now().strftime("%c"),
        city,i,len(file_list))
        print(log)
        file_path = os.path.join(city_folder,file_list[i])
        out_path = os.path.join(out_folder,file_list[i].replace(".npy",".png"))
        emulmap_np = np.load(file_path)
        edge_color = visulize(emulmap_np)
        cv2.imwrite(out_path,edge_color)



if __name__ == "__main__":
    root_path = "../kitti-semantic/data_semantics/training"

    vis_kitti(root_path)


