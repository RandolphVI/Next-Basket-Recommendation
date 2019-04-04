# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import random
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def split_data(input_file):
    """
    Split_data.
    Args:
        input_file:
    Returns:
        Train Valid Test

    0-11 个时间戳的 items 用于训练 12 时刻的 items 用于测试
    每个时刻的 items 称为一个 basket

    示例：
    userID  baskets   num_baskets
    1   [[2, 5],[3, 7]]  2
    """
    data = pd.read_csv(input_file, names=['userID', 'itemID', 'timestamp'])
    baskets = data.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  #
    baskets.columns = ['userID', 'baskets']
    baskets['num_baskets'] = baskets.baskets.apply(len)
    users = baskets.userID.values
    train_valid = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:-1]
        row = [user, b, baskets.iloc[index]['num_baskets'] - 1]
        train_valid.append(row)
    train_set, valid_set = train_test_split(train_valid, test_size=0.1)  # 留出 10% 验证集
    train_set = pd.DataFrame(train_set, columns=['userID', 'baskets', 'num_baskets'])
    valid_set = pd.DataFrame(valid_set, columns=['userID', 'baskets', 'num_baskets'])
    test_set = []
    for user in train_set.userID.values:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][-1]
        row = [user, b]
        test_set.append(row)
    test_set = pd.DataFrame(test_set, columns=['userID', 'baskets'])

    train_set.to_json('../data/train.json', orient='records', lines=True)
    test_set.to_json('../data/test.json', orient='records', lines=True)
    valid_set.to_json('../data/validation.json', orient='records', lines=True)

    return train_set, test_set, valid_set


def negative_sample(pickle_file):
    """
    Create negative sample.
    所有的 items 减去用户交互过的 items 为负样本 这个过程有点慢

    Args:
        pickle_file: 存储文件
    Returns:
        字典存储 (key: values) -> (userID: userID 的负样本列表)
    """
    with open(pickle_file, 'wb') as handle:
        train_data = pd.read_json('../data/train.json', orient='records', lines=True)
        valid_data = pd.read_json('../data/validation.json', orient='records', lines=True)
        total_data = train_data.append(valid_data)
        neg_samples = {}
        total_items = set()
        users = total_data.userID.values
        time_baskets = total_data.baskets.values

        for baskets in tqdm(time_baskets):
            for basket in baskets:
                for item in basket:
                    total_items.add(item)

        for u in tqdm(users):
            history_items = set()
            u_baskets = total_data[total_data['userID'] == u].baskets.values  # user 历史交互记录
            for basket in u_baskets[0]:
                for item in basket:
                    history_items.add(item)
            neg_items = total_items - history_items
            neg_samples[u] = neg_items
        pickle.dump(neg_samples, handle)


if __name__ == '__main__':
    # train, test, valid = split_data('../data/ratingT12.csv')
    # print(train[0:5].baskets.values)
    # print('----------')
    negative_sample('../data/neg_sample.pickle')

