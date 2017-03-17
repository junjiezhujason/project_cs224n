#!/bin/bash

cl upload code
cl upload data
cl upload train

# cl run --name run-predict --request-docker-image sckoo/cs224n-squad:v4 :code :data :train dev.json:0x4870af2556994b0687a1927fcec66392 'python code/qa_answer.py --dev_path dev.json'
# specialized command
cl run --name run-predict --request-docker-image sckoo/cs224n-squad:v4 :code :data :train dev.json:0x4870af2556994b0687a1927fcec66392 'python code/qa_answer.py --dev_path dev.json --batch_size=32 --max_context_length=766 --model==matchLSTM --decoder_type=naive --state_size=150'

