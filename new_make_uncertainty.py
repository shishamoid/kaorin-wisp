from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import statistics
#import mathf
from sklearn.preprocessing import MinMaxScaler

noise_dim_list = [40]
noise_size_list = [10,100,200,500]
#img_idx_list = ["0035","0097"]
img_idx_list = ["0035","0003","0097"]
inval_or_notinval = "notinval"

def to_pil_image(image):
    return Image.fromarray((image*255).astype(dtype=np.uint8))

def get_image_list(path_list): 
    image_array_all = []
    print("len(path_list)",len(path_list))

    for i in range(len(path_list)):
        #print("パス",path_list[i])
        image_1 = Image.open(path_list[i][0])#globがリストでとってくる
        image_array = np.array(image_1)
        image_array = image_array.astype(np.float64) / 255
        image_array_all.append(image_array)
    return image_array_all

for k in range(len(img_idx_list)):
    image_path_list = []
    uncer_list = np.zeros((1920,1080))
    img_idx = img_idx_list[k]
    for j in range(len(noise_size_list)):
        noise_size = noise_size_list[j]

        for i in range(len(noise_dim_list)):
            noise_dim = noise_dim_list[i]
            if inval_or_notinval == "inval":
                if noise_size in [1,4]:#/home/ito/kaolin-wisp/0123-0316_noise_dim40_noise_size500/val/0003-0-lod1.png
                    image_path_list.append(glob.glob("/home/ito/kaolin-wisp/uncertainly_map_inval/*_noise_dim{}_noise_size{}/val/{}-*-lod1.png".\
                    format(noise_dim,noise_size,img_idx)))
                elif noise_size in [10,100,200,500]:
                    image_path_list.append(glob.glob("/home/ito/kaolin-wisp/uncertainly_map_inval/*_noise_dim{}_noise_size{}/val/{}-*-lod1.png".\
                    format(noise_dim,noise_size,img_idx)))
            
            elif inval_or_notinval == "notinval":
                if noise_size in [1,4]:#/home/ito/kaolin-wisp/0123-0316_noise_dim40_noise_size500/val/0003-0-lod1.png
                    image_path_list.append(glob.glob("/home/ito/kaolin-wisp/uncertainly_map_notinval/*_noise_dim{}_noise_size{}/val/{}-*-lod1.png".\
                    format(noise_dim,noise_size,img_idx)))
                elif noise_size in [10,100,200,500]:
                    image_path_list.append(glob.glob("/home/ito/kaolin-wisp/uncertainly_map_notinval/*_noise_dim{}_noise_size{}/val/{}-*-lod1.png".\
                    format(noise_dim,noise_size,img_idx)))

    print("path_list",len(image_path_list))
    image_list = get_image_list(image_path_list)
    image_1 = image_list
    print("=======================")
    print("SHAPE",image_1[1][1][1][1])

    start = time.time()
    _uncer_list=np.zeros((1920,1080))
    for n in range(1920):
        for m in range(1080):
            _un_list_3 = np.zeros(1)

            for s in range(3):
            #np.mean([image_1[i][j][k],correct_image[i][j][]])
                #for i in range(len(image_1))
                #print()
                varia = np.var([[image_1[0][n][m][s]],[image_1[1][n][m][s]],[image_1[2][n][m][s]],[image_1[3][n][m][s]]])

                _un_list_3+=varia
            #print(_uncer_list[n][m])
            _uncer_list[n][m] = _un_list_3/3

    print(_uncer_list.shape)

    #uncer_list = np.sum(_uncer_list,axis=2)

    stop = time.time()
    print("時間",start-stop)

    #min_val = _uncer_list.min()
    max_val = _uncer_list.max()
    min_val = _uncer_list.min()
    #assert(not np.isclose(max_val - min_val, 0)) # ゼロ割防止のチェック
    _uncer_list = _uncer_list / (max_val - min_val)
    

    #rint(_uncer_list.shape)
    uncer_list = _uncer_list

    #uncer_list = uncer_list/len(noise_size_list)
    uncer_list_image = to_pil_image(uncer_list)
    uncer_list_image.save("./uncertainlymap_{}_{}_9.png".format(img_idx,inval_or_notinval))