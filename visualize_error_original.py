from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# numpy array -> PIL Imageの変換，何回も使うので
def to_pil_image(image):
    return Image.fromarray((image*255).astype(dtype=np.uint8))

#image_num = "0003"

image_num_list = ["0003","0035","0054","0074","0078","0081","0081","0089","0097","0107","0115"]

for i in range(len(image_num_list)):
    image_num = image_num_list[i]
    print(image_num)

    if image_num == "0035":
        val_num = "1"
    elif image_num == "0054":
        val_num = "2"
    elif image_num == "0003":
        val_num = "0"
    elif image_num == "0074":
        val_num = "3"
    elif image_num == "0078":
        val_num = "4"
    elif image_num == "0081":
        val_num = "5"
    elif image_num == "0089":
        val_num = "6"
    elif image_num == "0097":
        val_num = "7"
    elif image_num == "0107":
        val_num = "8"
    elif image_num == "0115":
        val_num = "9"

    # 正解の画像
    image_path_1 = Path.cwd() / "./fox/val/{}.png".format(image_num,val_num)

    # 他方の画像（比較画像を想定）
    image_path_2 = Path.cwd() / "./_results/logs/runs/test-ngp-nerf/20221103-145343/val/{}-{}-lod15.png".format(image_num,val_num)
    #image_path_2 = Path.cwd() / "./_results/logs/runs/test-ngp-nerf/20221021-020135/val/{}-*-lod15.png".format(image_num)
    
    # 存在チェック
    assert(image_path_1.exists())
    assert(image_path_2.exists())

    # 画像読み込み（ここではPillowを使っていますがOpenCVでも同様の処理はできます）
    image_1 = Image.open(image_path_1)
    image_2 = Image.open(image_path_2)

    # numpyのarrayに変換
    image_1 = np.array(image_1)
    image_2 = np.array(image_2)

    # uint8で読み込まれるはずなので扱いやすくfloatにしておく
    # 輝度値の範囲も[0, 255]から[0, 1]に
    image_1 = image_1.astype(np.float64) / 255
    image_2 = image_2.astype(np.float64) / 255

    # 画像サイズの確認，一致しているかチェック
    print(f"Image 1 Size: {image_1.shape[1]} x {image_1.shape[0]}, {image_1.shape[2]}ch")
    print(f"Image 2 Size: {image_2.shape[1]} x {image_2.shape[0]}, {image_2.shape[2]}ch")
    assert(image_1.shape == image_2.shape)

    # 誤差の例；各チャンネル絶対値誤差→チャンネルで平均
    absolute_errors = np.abs(image_1 - image_2).mean(axis=2)

    # 誤差ヒストグラムの可視化
    data = absolute_errors.flatten() # 一列にする
    fig = plt.figure() # ヒストグラム画像の保存，ちょっと行儀悪いけどとりあえず
    plt.hist(data, bins=16) # ヒストグラムを描画
    plt.ylabel("noise frequency")
    plt.xlabel("noise size")
    fig.savefig(Path.cwd() / "./visualize_error2/{}_absolute_errors_hist.png".format(image_num)) # 保存
    
    # 画像保存
    absolute_errors = to_pil_image(absolute_errors)
    absolute_errors.save(Path.cwd() / "./visualize_error2/{}_absolute_errors_image.png".format(image_num))

    # 他の誤差の例；各チャンネル絶対値誤差→チャンネルで平均→最大値・最小値で正規化
    absolute_errors = np.abs(image_1 - image_2).mean(axis=2)
    min_val = absolute_errors.min()
    max_val = absolute_errors.max()
    assert(not np.isclose(max_val - min_val, 0)) # ゼロ割防止のチェック
    normalized_absolute_errors = absolute_errors / (max_val - min_val)
    normalized_absolute_errors = to_pil_image(normalized_absolute_errors)
    #normalized_absolute_errors.show()

    #追加
    normalized_absolute_errors.save(Path.cwd() / "./visualize_error2/{}_normalized_absolute_errors_image.png".format(image_num))

    # 他の誤差の例；チャンネルについて平均二乗平方根誤差(RMSE)→最大値・最小値で正規化
    rmse_errors = np.sqrt(np.power(image_1 - image_2, 2).mean(axis=2))
    min_val = rmse_errors.min()
    max_val = rmse_errors.max()
    assert(not np.isclose(max_val - min_val, 0)) # ゼロ割防止のチェック
    normalized_rmse_errors = rmse_errors / (max_val - min_val)
    normalized_rmse_errors = to_pil_image(normalized_rmse_errors)
    #normalized_rmse_errors.show()

    #追加
    normalized_rmse_errors.save(Path.cwd() / "./visualize_error2/{}_normalized_rmse_errors_image.png".format(image_num))