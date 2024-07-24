#!/bin/bash

python main.py \
--train_list lists/FAME.txt \
--save_path exps/debug \
--lr 0.000010 \
--lr_decay 0.97 \
--test_step 1 \
--max_epoch 10 \
--n_cpu 12 \
--batch_size 400 \
--EU EU \
--pretrain_s 'ecapa' \
--pretrain_f 'face18' \
# --initial_model 'pretrain/seen.pt' \
# --eval