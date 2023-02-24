from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

noise_dim_list = [10,40,80,120]
noise_size_list = [50]
img_idx_list = ["0003","0035","0097"]

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

for k in range(len(img_idx_list)):
    img_idx = img_idx_list[k]
    for j in range(len(noise_size_list)):
        noise_size = noise_size_list[j]

        for i in range(len(noise_dim_list)):
            noise_dim = noise_dim_list[i]
            #_results/logs/runs/in_val
            #_results/logs/runs/test-ngp-nerf/old
            
            image_path_list = glob.glob("/home/ito/kaolin-wisp/_results/logs/runs/not_in_val/*_noise_dim{}_noise_size{}/val/{}-*-lod1.png".\
                format(noise_dim,noise_size,img_idx))
                    
            """
            image_path_list = glob.glob("/home/ito/kaolin-wisp/_results/logs/runs/test-ngp-nerf/old/*_noise_dim{}_noise_size{}/val/{}-*-lod1.png".\
                format(noise_dim,noise_size,img_idx))
            """
            correct_path = glob.glob("./fox_light/val/{}*".format(img_idx))

            correct_path = correct_path[0]
            
            #numpy array -> PIL Imageの変換，何回も使うので
           
            print("path_list",image_path_list)
            print("img_i","/home/ito/kaolin-wisp/_results/logs/runs/not_in_val/*_noise_dim{}_noise_size{}/val/{}-*-lod1.png".\
                format(noise_dim,noise_size,img_idx))
            
            image_1 = average_image(image_path_list)

            image_path_2 = correct_path
            image_2 = Image.open(image_path_2)
            image_2 = np.array(image_2)

            image_2 = image_2.astype(np.float64) / 255

            # 画像サイズの確認，一致しているかチェック
            print(f"Image 1 Size: {image_1.shape[1]} x {image_1.shape[0]}, {image_1.shape[2]}ch")
            print(f"Image 2 Size: {image_2.shape[1]} x {image_2.shape[0]}, {image_2.shape[2]}ch")
            assert(image_1.shape == image_2.shape)

            # 誤差の例；各チャンネル絶対値誤差→チャンネルで平均
            absolute_errors = np.abs(image_1 - image_2).mean(axis=2)
            data=absolute_errors
            # 誤差ヒストグラムの可視化
            data = absolute_errors.flatten() # 一列にする
            fig = plt.figure() # ヒストグラム画像の保存，ちょっと行儀悪いけどとりあえず
            plt.hist(data, bins=16) # ヒストグラムを描画
            plt.ylabel("difference frequency",fontsize=18)
            plt.xlabel("difference size",fontsize=18)
            plt.ylim(0, 1*10**6.5)
            plt.xlim(0, 1.5)
            fig.savefig(Path.cwd() / "./visualize_diff_notinval/{}_noisesize_{}_noisedim_{}_notinval.png".format(img_idx,noise_size,noise_dim)) # 保存
