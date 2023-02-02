from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import statistics
#import mathf

noise_dim_list = [40]
noise_size_list = [10,100,200,500]
img_idx_list = ["0035","0097"]

def to_pil_image(image):
    return Image.fromarray((image*255).astype(dtype=np.uint8))

def average_image(path_list): #同じ条件で10回学習しているので、その平均をとっています
    image_array_all = np.zeros((1920,1080,3), dtype=float, order='C')
    for i in range(len(path_list)):
        image_1 = Image.open(path_list[i])
        image_array = np.array(image_1)
        image_array = image_array.astype(np.float64) / 255
        image_array_all +=image_array

    average = image_array_all/(len(path_list))
    return average

def get_image_list(path_list): #同じ条件で10回学習しているので、その平均をとっています
    image_array_all = np.zeros((1920,1080,3), dtype=float, order='C')
    for i in range(len(path_list)):
        image_1 = Image.open(path_list[i])
        image_array = np.array(image_1)
        image_array = image_array.astype(np.float64) / 255
        image_array_all +=image_array*10
    return image_array_all

    
for k in range(len(img_idx_list)):
    uncer_list = np.zeros((1920,1080))
    img_idx = img_idx_list[k]
    for j in range(len(noise_size_list)):
        noise_size = noise_size_list[j]

        for i in range(len(noise_dim_list)):
            noise_dim = noise_dim_list[i]
            #_results/logs/runs/in_val
            #_results/logs/runs/test-ngp-nerf/old
            if noise_size in [1,4]:#/home/ito/kaolin-wisp/0123-0316_noise_dim40_noise_size500/val/0003-0-lod1.png
                image_path_list = glob.glob("/home/ito/kaolin-wisp/uncertainly_map_inval/*_noise_dim{}_noise_size{}/val/{}-*-lod1.png".\
                format(noise_dim,noise_size,img_idx))
            elif noise_size in [10,100,200,500]:
                image_path_list = glob.glob("/home/ito/kaolin-wisp/uncertainly_map_inval/*_noise_dim{}_noise_size{}/val/{}-*-lod1.png".\
                format(noise_dim,noise_size,img_idx))
            
        print("path_list",image_path_list)
            
        #simage_1 = average_image(image_path_list) #予測したやつ
        
        print("image_path_list",image_path_list)
        image_list = get_image_list(image_path_list)

        correct_path = glob.glob("./fox_light/val/{}*".format(img_idx))[0]
        correct_image = Image.open(correct_path)
        correct_image = np.array(correct_image)
        correct_image = correct_image.astype(np.float64) / 255

        """
        for i in range(len(image_list)):
            image_1 = image_list[i]
            absolute_errors = np.abs(image_1 - correct_image).mean(axis=2)
            print(absolute_errors.shape)
        """
        print(len(image_list))
        image_1 = image_list
        #absolute_errors = np.abs(image_1 - correct_image).mean(axis=2)
        print("=======================")
        #print(absolute_errors.shape)
        print("image_1",image_1.shape)
        print("correct_image",correct_image.shape)
        start = time.time()
        _uncer_list=np.zeros((1920,1080))
        for n in range(1920):
            for m in range(1080):
                _un_list_3 = np.zeros(1)

                for s in range(3):
                #np.mean([image_1[i][j][k],correct_image[i][j][]])
                    varia = np.var([image_1[n][m][s],correct_image[n][m][s]])
                    _un_list_3[0]+=varia
                #print("------------------------------")
                #print(absolute_errors[i][j])
                #print("varia",varia)
                #try:
                _uncer_list[n][m] = _un_list_3
        
        print(_uncer_list.shape)

        #uncer_list = np.sum(_uncer_list,axis=2)

        stop = time.time()
        print("時間",start-stop)

        #rint(_uncer_list.shape)
        uncer_list += _uncer_list
    
    uncer_list = uncer_list/len(noise_size_list)
    uncer_list_image = to_pil_image(uncer_list)
    uncer_list_image.save("./uncertainlymap_{}_inval_7.png".format(img_idx))