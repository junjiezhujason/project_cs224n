#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import pickle
import logging
from collections import Counter, defaultdict
import argparse

from tensorflow.python.platform import gfile
import numpy as np
from os.path import join as pjoin
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def load_glove_embeddings(glove_path):
    glove = np.load(glove_path)['glove']
    logger.info("Loading glove embedding")
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    logger.info("dtype of glove is: %s" % type(glove))
    logger.info("dtype of glove is: %s" % type(glove[0][0]))
    glove = tf.to_float(glove)
    logger.info("glove is: " + str(glove) )
    return glove

def load_dataset(source_dir, data_mode, max_q_toss, max_c_toss, data_pfx_list=None):

    assert os.path.exists(source_dir)

    train_pfx = pjoin(source_dir, "train")
    valid_pfx = pjoin(source_dir, "val")
    dev_pfx = pjoin(source_dir, "dev")

    if data_mode=="tiny":
        max_train = 100
        max_valid = 10
        max_dev = 100

    train = []
    valid = []
    train_raw = []
    valid_raw = []
    
    dev = []
    dev_raw = []

    max_c_len = 0
    max_q_len = 0
    
    if data_pfx_list == None:
        data_pfx_list = [train_pfx, valid_pfx]
    else:
        data_pfx_list = [pjoin(source_dir, data_pfx) for data_pfx in data_pfx_list]

    for data_pfx in data_pfx_list:
        if data_pfx == train_pfx:
            data_list = train
            data_list_raw = train_raw
            if data_mode=="tiny":
                max_entry = max_train
            logger.info("")
            logger.info("Loading training data")
        if data_pfx == valid_pfx:
            data_list = valid 
            data_list_raw = valid_raw
            if data_mode=="tiny":
                max_entry = max_valid
            logger.info("")
            logger.info("Loading validation data")
        if data_pfx == dev_pfx:
            data_list = dev
            data_list_raw = dev_raw
            if data_mode=="tiny":
                max_entry = max_dev
            logger.info("")
            logger.info("Loading as dev data")

        c_ids_path = data_pfx + ".ids.context"
        c_raw_path = data_pfx + ".context" 
        q_ids_path = data_pfx + ".ids.question"
        q_raw_path = data_pfx + ".question" 
        label_path = data_pfx + ".span"

        counter = 0
        ignore_counter= 0

        uuid_list = []
        if data_pfx == dev_pfx:
            uuid_path = data_pfx + ".uuid"       
            with gfile.GFile(uuid_path, mode="rb") as uuid_file:
                for line in uuid_file:
                    uuid_list.append(line.strip())

        with gfile.GFile(q_raw_path, mode="rb") as r_q_file:
            with gfile.GFile(c_raw_path, mode="rb") as r_c_file:
                with gfile.GFile(q_ids_path, mode="rb") as q_file:
                    with gfile.GFile(c_ids_path, mode="rb") as c_file:
                        with gfile.GFile(label_path, mode="rb") as l_file:
                            for line in l_file:
                                label = map(int,line.strip().split(" "))
                                context = map(int, c_file.readline().strip().split(" "))
                                question = map(int,q_file.readline().strip().split(" "))
                                context_raw = r_c_file.readline().strip().split(" ")
                                question_raw = r_q_file.readline().strip().split(" ")

                                c_len = len(context)
                                q_len = len(question)

                                # Do not toss out, only  truncate for dev set
                                if q_len > max_q_toss:
                                    if data_pfx == dev_pfx:
                                        q_len = max_q_toss
                                        question = question[:max_q_toss]
                                    else:
                                        ignore_counter += 1
                                        continue
                                if c_len > max_c_toss:
                                    if data_pfx == dev_pfx:
                                        c_len = max_c_toss
                                        context = context[:max_c_toss]
                                    else:
                                        ignore_counter += 1
                                        continue

                                max_c_len = max(max_c_len, c_len)
                                max_q_len = max(max_q_len, q_len)

                                entry = [question, q_len, context, c_len, label]
                                data_list.append(entry)
                                
                                raw_entry = [question_raw, context_raw]
                                data_list_raw.append(raw_entry)

                                counter += 1
                                if counter % 10000 == 0:
                                    logger.info("read %d context lines" % counter)
                                if data_mode=="tiny": 
                                    if counter==max_entry:
                                        break

        logger.info("Ignored %d questions/contexts in total" % ignore_counter)
        assert counter>0, "No questions/contexts left (likely filtered out)"

        logger.info("read %d questions/contexts in total" % counter)
        logger.info("maximum question length %d" % max_q_len)
        logger.info("maximum context length %d" % max_c_len)

    dataset = {"training":train, "validation":valid, "training_raw":train_raw, "validation_raw":valid_raw, "dev":dev, "dev_raw":dev_raw, "dev_uuid":uuid_list}
    return dataset, max_q_len, max_c_len

def summarize_dataset(source_dir, out_dir, data_type="train"):

    assert os.path.exists(source_dir)
    assert os.path.exists(out_dir)
    logger.info("Loading data")

    data_pfx = pjoin(source_dir, data_type)
    out_pfx = pjoin(out_dir, data_type)

    # c_ids_path = data_pfx + ".ids.context"
    c_raw_path = data_pfx + ".context" 
    # q_ids_path = data_pfx + ".ids.question"
    q_raw_path = data_pfx + ".question" 
    label_path = data_pfx + ".span"

    counter = 0

    len_entry = []
    first_word_dict = defaultdict(int) 

    out_len_path = out_pfx + ".length"
    our_first_word_path = out_pfx + ".firstword"

    with gfile.GFile(out_len_path, mode="w") as out_file: 
        with gfile.GFile(q_raw_path, mode="rb") as q_file:
            with gfile.GFile(c_raw_path, mode="rb") as c_file:
                with gfile.GFile(label_path, mode="rb") as l_file:
                    for line in l_file:
                        # compute label length
                        label = map(int,line.strip().split(" "))
                        a_len = label[1]-label[0]+1

                        c_text = c_file.readline().strip().split(" ")
                        q_text = q_file.readline().strip().split(" ")
                        # print(q_text[0])

                        first_word_dict[q_text[0]] += 1

                        c_len = len(c_text)
                        q_len = len(q_text)

                        out_file.write("\t".join([str(q_len), str(c_len), str(a_len)])+"\n")
                        
                        counter += 1
                        if counter % 10000 == 0:
                            logger.info("read %d context lines" % counter)

    min_freq = 100
    for key in first_word_dict:
        if first_word_dict[key] > min_freq :
            print(key+": "+str(first_word_dict[key]))


if __name__=="__main__":
    data_dir="data/squad"
    out_dir ="summary"
    summarize_dataset(data_dir, out_dir)
    # summarize_dataset(data_dir, out_dir, data_type="val")
    
