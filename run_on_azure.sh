#!/bin/bash


NAME=exp0
MDIR=run/${NAME}
LDIR=${MDIR}/log
TDIR=${MDIR}/train

source .env/bin/activate

rm -r train/
rm -r data/tmp-squad-train 
rm -r log/

# create folder for experiment
mkdir -p ${MDIR} 

# run python script
python code/train.py --model=baseline \
                     --decoder_type=naive \
                     --data_size=full \
                     --batch_size=32 \
                     --learning_rate=0.03 \
                     --state_size=50 \ 
                     --log_dir=${LDIR} \
                     --train_dir=${TDIR} 
