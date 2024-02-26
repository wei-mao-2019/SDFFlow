#!/bin/bash

#ROOT=/home/wei/Documents/projects/2022-nerf/SDFFlow
#SCAN=ian3
##EXPNAME=_$SCAN
##TIMESTAMP=2023_07_26_13_53_06_weights
##CKPT=2000
#GPU=0
#
### plot
#CUDA_VISIBLE_DEVICES=$GPU python training/exp_runner.py --conf /home/wei/Documents/projects/2022-nerf/SDFFlow/code/confs/pretrain_flow.conf --gpu 0 --nepoch 2000 --is_pretrain_flow


ROOT=/home/wei/Documents/projects/2022-nerf/SDFFlow
GPU=0

#python training/exp_runner.py --conf ./confs/total_capture.conf --gpu 0 --nepoch 2000 \
#     --is_dist --gpus 2 --cancel_vis --is_only_plot

SCAN=ian3
EXPNAME=_$SCAN
TIMESTAMP=2023_11_29_17_10_04
CKPT=2000

## plot
#python training/exp_runner.py --conf $ROOT/exps/total_capture$EXPNAME/$TIMESTAMP/runconf.conf --gpu 0 --nepoch 2000 \
#       --cancel_vis --is_only_plot --is_continue --checkpoint $CKPT

python evaluation/eval_flow_geo.py --conf $ROOT/exps/total_capture$EXPNAME/$TIMESTAMP/runconf.conf \
       --checkpoint $CKPT --timestamp $TIMESTAMP



