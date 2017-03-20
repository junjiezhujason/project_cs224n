#!/bin/bash

EXP=exp009

# cl upload code
# cl upload data 
# cl upload $EXP

cl run --name run-predict --request-docker-image sckoo/cs224n-squad:v4 :code :data :exp009 dev.json:0x4870af2556994b0687a1927fcec66392 'python code/qa_answer.py --dev_path dev.json --config_path=exp009/log/flags.json --train_dir=exp009/train'

