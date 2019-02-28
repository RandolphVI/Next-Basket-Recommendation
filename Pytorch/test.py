# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import time
import random
import math
import pickle
import torch
import numpy as np
from config import Config
from utils import data_helpers as dh

logger = dh.logger_fn("torch-log", "logs/test-{0}.log".format(time.asctime()))

MODEL = input("☛ Please input the model file you want to test: ")

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("✔︎ The format of your input is legal, now loading to next step...")

MODEL_DIR = dh.load_model_file(MODEL)


def test():
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    train_data = dh.load_data(Config().TRAININGSET_DIR)

    logger.info("✔︎ Test data processing...")
    test_data = dh.load_data(Config().TESTSET_DIR)

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Load model
    dr_model = torch.load(MODEL_DIR)

    dr_model.eval()

    item_embedding = dr_model.encode.weight
    hidden = dr_model.init_hidden(Config().batch_size)

    hitratio_numer = 0
    hitratio_denom = 0
    ndcg = 0.0

    for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=False)):
        uids, baskets, lens = x
        dynamic_user, _ = dr_model(baskets, lens, hidden)
        for uid, l, du in zip(uids, lens, dynamic_user):
            scores = []
            du_latest = du[l - 1].unsqueeze(0)

            # calculating <u,p> score for all test items <u,p> pair
            positives = test_data[test_data['userID'] == uid].baskets.values[0]  # list dim 1
            p_length = len(positives)
            positives = torch.LongTensor(positives)

            # Deal with positives samples
            scores_pos = list(torch.mm(du_latest, item_embedding[positives].t()).data.numpy()[0])
            for s in scores_pos:
                scores.append(s)

            # Deal with negative samples
            negtives = random.sample(list(neg_samples[uid]), Config().neg_num)
            negtives = torch.LongTensor(negtives)
            scores_neg = list(torch.mm(du_latest, item_embedding[negtives].t()).data.numpy()[0])
            for s in scores_neg:
                scores.append(s)

            # Calculate hit-ratio
            index_k = []
            for k in range(Config().top_k):
                index = scores.index(max(scores))
                index_k.append(index)
                scores[index] = -9999
            hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
            hitratio_denom += p_length

            # Calculate NDCG
            u_dcg = 0
            u_idcg = 0
            for k in range(Config().top_k):
                if index_k[k] < p_length:  # 长度 p_length 内的为正样本
                    u_dcg += 1 / math.log(k + 1 + 1, 2)
                u_idcg += 1 / math.log(k + 1 + 1, 2)
            ndcg += u_dcg / u_idcg

    hitratio = hitratio_numer / hitratio_denom
    ndcg = ndcg / len(train_data)
    print('Hit ratio[{0}]: {1}'.format(Config().top_k, hitratio))
    print('NDCG[{0}]: {1}'.format(Config().top_k, ndcg))


if __name__ == '__main__':
    test()


