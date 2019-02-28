# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import random
import logging
import pickle
import json
import torch
import numpy as np
import pandas as pd


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def load_data(input_file, flag=None):
    if flag:
        data = pd.read_json(input_file, orient='records', lines=True)
    else:
        data = pd.read_json(input_file, orient='records', lines=True)

    return data


def load_model_file(checkpoint_dir):
    MODEL_DIR = 'runs/' + checkpoint_dir
    names = [name for name in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, name))]
    max_epoch = 0
    choose_model = ''
    for name in names:
        if int(name[6:8]) >= max_epoch:
            max_epoch = int(name[6:8])
            choose_model = name
    MODEL_FILE = 'runs/' + checkpoint_dir + '/' + choose_model
    return MODEL_FILE


def sort_batch_of_lists(uids, batch_of_lists, lens):
    """Sort batch of lists according to len(list). Descending"""
    sorted_idx = [i[0] for i in sorted(enumerate(lens), key=lambda x: x[1], reverse=True)]
    uids = [uids[i] for i in sorted_idx]
    lens = [lens[i] for i in sorted_idx]
    batch_of_lists = [batch_of_lists[i] for i in sorted_idx]
    return uids, batch_of_lists, lens


def pad_batch_of_lists(batch_of_lists, pad_len):
    """Pad batch of lists."""
    padded = [l + [[0]] * (pad_len - len(l)) for l in batch_of_lists]
    return padded


def batch_iter(data, batch_size, pad_len, shuffle=True):
    """
    Turn dataset into iterable batches.

    Args:
        data: The data
        batch_size: The size of the data batch
        pad_len: The padding length
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data_size = len(data)
    num_batches_per_epoch = data_size // batch_size
    if shuffle:
        shuffled_data = data.sample(frac=1)
    else:
        shuffled_data = data

    for i in range(num_batches_per_epoch):
        uids = shuffled_data[i * batch_size: (i + 1) * batch_size].userID.values
        baskets = list(shuffled_data[i * batch_size: (i + 1) * batch_size].baskets.values)
        lens = shuffled_data[i * batch_size: (i + 1) * batch_size].num_baskets.values
        uids, baskets, lens = sort_batch_of_lists(uids, baskets, lens)  # sort 排序是为了后序 pack_packed_sequence() 函数
        baskets = pad_batch_of_lists(baskets, pad_len)
        yield uids, baskets, lens

    if data_size % batch_size != 0:
        # 将最后一个不满 batch_size 的 batch 进行随机补齐
        residual = [i for i in range(num_batches_per_epoch * batch_size, data_size)] + list(
            np.random.choice(data_size, batch_size - data_size % batch_size))
        uids = shuffled_data.iloc[residual].userID.values
        baskets = list(shuffled_data.iloc[residual].baskets.values)
        lens = shuffled_data.iloc[residual].num_baskets.values
        uids, baskets, lens = sort_batch_of_lists(uids, baskets, lens)
        baskets = pad_batch_of_lists(baskets, pad_len)
        yield uids, baskets, lens


def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]


def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)
