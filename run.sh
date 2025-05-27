#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=BraTS18_[80,160,160]_SGD_b2_lr-2_multiTeacher_wt.1

CUDA_VISIBLE_DEVICES=$1 python train.py \
--snapshot_dir=snapshots/$name/ \
--input_size=80,160,160 \
--batch_size=2 \
--num_gpus=1 \
--num_steps=80000 \
--val_pred_every=5000 \
--learning_rate=1e-2 \
--num_classes=3 \
--num_workers=4 \
--train_list=BraTS18/BraTS18_train_all.csv \
--val_list=BraTS18/BraTS18_val.csv \
--random_mirror=True \
--random_scale=True \
--weight_std=True \
--reload_path=snapshots/NA/last.pth \
--reload_from_checkpoint=False > logs/${time}_train_${name}.log 2>&1 &


time=$(date "+%Y%m%d-%H%M%S")
name=Random_Missing_BraTS18_[80,160,160]_SGD_b2_lr-2_multiTeacher_wt.1

CUDA_VISIBLE_DEVICES=$1 python train.py \
--snapshot_dir=snapshots/$name/ \
--input_size=80,160,160 \
--batch_size=2 \
--num_gpus=1 \
--num_steps=115000 \
--val_pred_every=5000 \
--learning_rate=1e-2 \
--num_classes=3 \
--num_workers=4 \
--train_list=BraTS18/BraTS18_train_all.csv \
--val_list=BraTS18/BraTS18_val.csv \
--random_mirror=True \
--random_scale=True \
--weight_std=True \
--reload_path=snapshots/BraTS18_[80,160,160]_SGD_b2_lr-2_multiTeacher_wt.1/final.pth \
--reload_from_checkpoint=True \
--mode=random > logs/${time}_train_${name}.log 2>&1 &
