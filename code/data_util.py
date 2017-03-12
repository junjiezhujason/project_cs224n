#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import pickle
import logging
from collections import Counter
import argparse

from tensorflow.python.platform import gfile
import numpy as np
from os.path import join as pjoin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def load_glove_embeddings(glove_path):
    glove = np.load(glove_path)['glove']
    logger.info("Loading glove embedding")
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    return glove

def load_dataset(source_dir, data_mode):

    assert os.path.exists(source_dir)

    train_pfx = pjoin(source_dir, "train")
    valid_pfx = pjoin(source_dir, "val")

    if data_mode=="tiny":
        max_train = 20
        max_valid = 10

    train = []
    valid = []
    train_raw = []
    valid_raw = []
    max_c_len = 0
    max_q_len = 0

    for data_pfx in train_pfx, valid_pfx:
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

        c_ids_path = data_pfx + ".ids.context"
        c_raw_path = data_pfx + ".context" 
        q_ids_path = data_pfx + ".ids.question"
        q_raw_path = data_pfx + ".question" 
        label_path = data_pfx + ".span"

        counter = 0

        with gfile.GFile(q_raw_path, mode="rb") as r_q_file:
            with gfile.GFile(c_raw_path, mode="rb") as r_c_file:
                with gfile.GFile(q_ids_path, mode="rb") as q_file:
                    with gfile.GFile(c_ids_path, mode="rb") as c_file:
                        with gfile.GFile(label_path, mode="rb") as l_file:
                            for line in l_file:
                                label = map(int,line.strip().split(" "))
                                context = map(int, c_file.readline().strip().split(" "))
                                question = map(int,q_file.readline().strip().split(" "))
                                entry = [question, len(question), context, len(context), label]
                                data_list.append(entry)
                                
                                context_raw = r_c_file.readline().strip().split(" ")
                                question_raw = r_q_file.readline().strip().split(" ")
                                raw_entry = [question_raw, context_raw]
                                data_list_raw.append(raw_entry)

                                max_c_len = max(max_c_len, len(context))
                                max_q_len = max(max_q_len, len(question))
                                counter += 1
                                if counter % 10000 == 0:
                                    logger.info("read %d context lines" % counter)
                                if data_mode=="tiny": 
                                    if counter==max_entry:
                                        break
        logger.info("read %d questions in total" % counter)
        logger.info("maximum question length %d" % max_q_len)
        logger.info("maximum context length %d" % max_c_len)

    dataset = {"training":train, "validation":valid, "training_raw":train_raw, "validation_raw":valid_raw}
    return dataset, max_q_len, max_c_len
