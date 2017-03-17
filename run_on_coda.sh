#!/bin/bash

cl upload code
cl upload data 
cl upload train

cl run --name run-predict --request-docker-image sckoo/cs224n-squad:v4 :code :data :train dev.json:0x4870af2556994b0687a1927fcec66392 'python code/qa_answer.py --dev_path dev.json --state_size=100 --model=baseline --batch_size=20'
