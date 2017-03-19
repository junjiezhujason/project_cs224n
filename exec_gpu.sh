#!/bin/bash

NAME=exp015
MDIR=run/${NAME}
LDIR=${MDIR}/log
TDIR=${MDIR}/train

# source .env/bin/activate
# rm -r data/tmp-squad-train 
# rm -r train/
# rm -r log/

# create folder for experiment
mkdir -p ${MDIR} 

export CUDA_VISIBLE_DEVICES=2
python code/train.py --log_dir=${LDIR} \
                     --train_dir=${TDIR} \
                     --data_size=full \
                     --model=matchLSTM \
                     --decoder_type=naive \
                     --epochs=15 \
                     --batch_size=64 \
                     --learning_rate=0.005 \
                     --state_size=150 \
                     --gpu_fraction=1.0 \
                     --num_epochs_per_decay=7 \

