#!/bin/bash

NAME=exp023
MDIR=run/${NAME}
LDIR=${MDIR}/log
TDIR=${MDIR}/train

# source .env/bin/activate
# rm -r data/tmp-squad-train 
# rm -r train/
# rm -r log/

# create folder for experiment
mkdir -p ${MDIR} 

export CUDA_VISIBLE_DEVICES=1
python code/train.py --log_dir=${LDIR} \
                     --train_dir=${TDIR} \
                     --data_size=full \
                     --model=matchLSTM \
                     --decoder_type=pointer \
                     --epochs=15 \
                     --batch_size=64 \
                     --learning_rate=0.001 \
                     --state_size=150 \
                     --dropout_keep_prob=0.85 \
                     --num_epochs_per_decay=7 \
                     --gpu_fraction=0.5 \

