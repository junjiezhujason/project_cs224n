#!/bin/bash

EXP=exp005

# cl work main::cs224n-jyj
cl upload code
# cl upload data 
cl upload $EXP

cl run --name run-predict --request-docker-image sckoo/cs224n-squad:v4 :code :data :exp005 dev.json:0x4870af2556994b0687a1927fcec66392 'python code/qa_answer.py --dev_path dev.json --config_path=exp005/log/flags.json --train_dir=exp005/train'

