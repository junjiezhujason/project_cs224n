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

    train = {}
    valid = {}

    for data_pfx in train_pfx, valid_pfx:
        if data_pfx == train_pfx:
            data_dict = train
            max_entry = max_train
            logger.info("Loading training data")
        if data_pfx == valid_pfx:
            data_dict = valid 
            max_entry = max_valid
            logger.info("Loading validation data")

        c_ids_path = data_pfx + ".ids.context"
        q_ids_path = data_pfx + ".ids.question"
        label_path = data_pfx + ".span"
        data_dict["question"] = []
        data_dict["context"] = []
        data_dict["label"] = []

        # load question data
        max_q_len = 0
        with gfile.GFile(q_ids_path, mode="rb") as data_file:
            counter = 0
            for line in data_file:
                counter += 1
                question = map(int,line.strip().split(" "))

                data_dict["question"].append(question)

                max_q_len = max(max_q_len, len(question))

                if counter % 10000 == 0:
                    logger.info("read %d context lines" % counter)
                if data_mode=="tiny" and counter==max_entry:
                    break
        logger.info("read %d questions in total" % counter)
        logger.info("maximum question length %d" % max_q_len)

        # load context data
        max_c_len = 0
        with gfile.GFile(c_ids_path, mode="rb") as data_file:
            counter = 0
            for line in data_file:
                counter += 1
                context = map(int, line.strip().split(" "))
                data_dict["context"].append(context)
                max_c_len = max(max_c_len, len(context))
                if counter % 10000 == 0:
                    logger.info("read %d question lines" % counter)
                if data_mode=="tiny" and counter==max_entry:
                    break
        logger.info("read %d contexts in total" % counter)
        logger.info("maximum context length %d" % max_c_len)

        # load labels
        with gfile.GFile(label_path, mode="rb") as data_file:
            counter = 0
            for line in data_file:
                counter += 1
                label = map(int,line.strip().split(" "))

                data_dict["label"].append(label)

                if counter % 10000 == 0:
                    logger.info("read %d context lines" % counter)
                if data_mode=="tiny" and counter==max_entry:
                    break

    dataset = {"training":train, "validation":valid}
        
    return dataset


def do_test(args):
    train, valid = load_id_data(args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('test', help='')	
    command_parser.add_argument('-d', '--data_mode', default="full", help="full: entire data set; tiny: first 10-20 entries")
    command_parser.add_argument('-s', '--source_dir', default="data/squad", help="Path to data")
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
