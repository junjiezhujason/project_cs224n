#!/bin/bash

cd /home/cs224n/project_cs224n

source .env/bin/activate

# python code/train.py

rm -r train/ 
rm -r log/ 
rm data/tmp-squad-train 

python code/train.py --model=matchLSTM --datasize=full --batch_size=32
