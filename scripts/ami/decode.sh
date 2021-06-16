#!/bin/bash
# Run decode.py with GPU

cmd="utils/queue.pl --mem 8G --gpu 1 --config conf/gpu.conf"

${cmd} decode.log \
  bash -c 'export CUDA_VISIBLE_DEVICES=$(free-gpu); /home/draj/anaconda3/envs/k2/bin/python ./decode.py'
