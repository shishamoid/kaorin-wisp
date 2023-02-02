#!/bin/bash
source `which virtualenvwrapper.sh`
workon wisp

noise_size="10"
noise_dim="40"

pretrained="/home/ito/kaolin-wisp/_results/logs/runs/in_val/1220-1103_noise_dim40_noise_size50/model.pth"

python app/main.py \
 --config configs/ngp_nerf.yaml \
 --multiview-dataset-format standard \
 --mip 0 \
 --dataset-path ./fox_light \
 --valid-only \
 --pretrained "$pretrained" \
 --noise_dim "$noise_dim" \
 --noise_size "$noise_size" \
 --log-dir_noise_dim "$noise_dim" \
 --log-dir_noise_size "$noise_size" \
 --log-dir ./uncertainly_map_inval/

noise_size="100"

 
python app/main.py \
 --config configs/ngp_nerf.yaml \
 --multiview-dataset-format standard \
 --mip 0 \
 --dataset-path ./fox_light \
 --valid-only \
 --pretrained "$pretrained" \
 --noise_dim "$noise_dim" \
 --noise_size "$noise_size" \
 --log-dir_noise_dim "$noise_dim" \
 --log-dir_noise_size "$noise_size" \
 --log-dir ./uncertainly_map_inval/

noise_size="200"

python app/main.py \
 --config configs/ngp_nerf.yaml \
 --multiview-dataset-format standard \
 --mip 0 \
 --dataset-path ./fox_light \
 --valid-only \
 --pretrained "$pretrained" \
 --noise_dim "$noise_dim" \
 --noise_size "$noise_size" \
 --log-dir_noise_dim "$noise_dim" \
 --log-dir_noise_size "$noise_size" \
 --log-dir ./uncertainly_map_inval/

noise_size="500"

python app/main.py \
 --config configs/ngp_nerf.yaml \
 --multiview-dataset-format standard \
 --mip 0 \
 --dataset-path ./fox_light \
 --valid-only \
 --pretrained "$pretrained" \
 --noise_dim "$noise_dim" \
 --noise_size "$noise_size" \
 --log-dir_noise_dim "$noise_dim" \
 --log-dir_noise_size "$noise_size" \
 --log-dir ./uncertainly_map_inval/

noise_size="10"
noise_dim="40"

pretrained="/home/ito/kaolin-wisp/_results/logs/runs/not_in_val/1228-0430_noise_dim40_noise_size50/model.pth"

python app/main.py \
 --config configs/ngp_nerf.yaml \
 --multiview-dataset-format standard \
 --mip 0 \
 --dataset-path ./fox_light \
 --valid-only \
 --pretrained "$pretrained" \
 --noise_dim "$noise_dim" \
 --noise_size "$noise_size" \
 --log-dir_noise_dim "$noise_dim" \
 --log-dir_noise_size "$noise_size" \
 --log-dir ./uncertainly_map_notinval/

noise_size="100"

python app/main.py \
 --config configs/ngp_nerf.yaml \
 --multiview-dataset-format standard \
 --mip 0 \
 --dataset-path ./fox_light \
 --valid-only \
 --pretrained "$pretrained" \
 --noise_dim "$noise_dim" \
 --noise_size "$noise_size" \
 --log-dir_noise_dim "$noise_dim" \
 --log-dir_noise_size "$noise_size" \
 --log-dir ./uncertainly_map_notinval/


noise_size="200"

python app/main.py \
 --config configs/ngp_nerf.yaml \
 --multiview-dataset-format standard \
 --mip 0 \
 --dataset-path ./fox_light \
 --valid-only \
 --pretrained "$pretrained" \
 --noise_dim "$noise_dim" \
 --noise_size "$noise_size" \
 --log-dir_noise_dim "$noise_dim" \
 --log-dir_noise_size "$noise_size" \
 --log-dir ./uncertainly_map_notinval/


noise_size="500"

python app/main.py \
 --config configs/ngp_nerf.yaml \
 --multiview-dataset-format standard \
 --mip 0 \
 --dataset-path ./fox_light \
 --valid-only \
 --pretrained "$pretrained" \
 --noise_dim "$noise_dim" \
 --noise_size "$noise_size" \
 --log-dir_noise_dim "$noise_dim" \
 --log-dir_noise_size "$noise_size" \
 --log-dir ./uncertainly_map_notinval/

