#!/bin/bash
source `which virtualenvwrapper.sh`
workon wisp

noise_size="50"
noise_dim="120"

for ((i=0 ; i<10 ; i++))
do
    date
    python3 app/main.py \
    --config configs/ngp_nerf.yaml \
    --multiview-dataset-format standard \
    --mip 0 \
    --dataset-path ./fox_light_include_val \
    --noise_dim "$noise_dim" \
    --noise_size "$noise_size" \
    --log-dir_noise_dim "$noise_dim" \
    --log-dir_noise_size "$noise_size"
done