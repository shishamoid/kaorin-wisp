#!/bin/bash
source `which virtualenvwrapper.sh`
workon wisp

noise_size="10"
noise_dim="10"
pretrained="/home/ito/kaolin-wisp/_results/logs/runs/not_in_val/0105-1055_noise_dim10_noise_size150/model.pth"

python3 app/main.py \
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
 --log-dir ./uncertainly_map/
    
noise_size="100"
noise_dim="100"

python3 app/main.py \
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
 --log-dir ./uncertainly_map/

noise_size="200"
noise_dim="10"

python3 app/main.py \
 --config configs/ngp_nerf.yaml \
 --multiview-dataset-format standard \
 --mip 0 \
 --dataset-path ./fox_light \
 --valid-only \
 --pretrained "$pretrained"\
 --noise_dim "$noise_dim" \
 --noise_size "$noise_size" \
 --log-dir_noise_dim "$noise_dim" \
 --log-dir_noise_size "$noise_size" \
 --log-dir ./uncertainly_map/
