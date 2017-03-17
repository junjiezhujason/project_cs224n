#!/bin/bash

source .env/bin/activate

rm -r train/
rm -r data/tmp-squad-train 
rm -r log/
python code/train.py --model=baseline \
                     --decoder_type=naive \
                     --max_context_length=300 \
                     --max_question_length=60 \
                     --data_size=full \
                     --batch_size=20 \
                     --learning_rate=0.03 \
                     --state_size=100
