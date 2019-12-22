# Deep Learning for Next Basket Recommendation

[![Python Version](https://img.shields.io/badge/language-python3.6-blue.svg)](https://www.python.org/downloads/) [![Build Status](https://travis-ci.org/RandolphVI/Next-Basket-Recommendation.svg?branch=master)](https://travis-ci.org/RandolphVI/Next-Basket-Recommendation) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/e704710bf363474d9ef334fd99367216)](https://www.codacy.com/app/chinawolfman/Next-Basket-Recommendation?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RandolphVI/Next-Basket-Recommendation&amp;utm_campaign=Badge_Grade) [![License](https://img.shields.io/github/license/RandolphVI/Next-Basket-Recommendation.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Issues](https://img.shields.io/github/issues/RandolphVI/Next-Basket-Recommendation.svg)](https://github.com/RandolphVI/Next-Basket-Recommendation/issues)

This repository contains my implementations of [DREAM](http://www.nlpr.ia.ac.cn/english/irds/People/sw/DREAM.pdf) for next basket prediction.

## Requirements

- Python 3.6
- Pytorch 0.4 +
- Pandas 0.23 +
- scikit-learn 0.19 +
- Numpy
- Gensim

## Data

You can download the [Negative Sample (neg_sample.pickle)](https://drive.google.com/open?id=19SnHlic2DswgyPwr9ul3t9uDsgEsaH1X) used in code. **Make sure they are under the `/data` folder.**

### Data Format

See data format in `data` folder which including the data sample files.

This repository can be used in other e-commerce datasets in two ways:
1. Modify your datasets into the same format of the sample.
2. Modify the data preprocess code in `data_helpers.py`.

Anyway, it should depend on what your data and task are.

## Network Structure

DREAM uses RNN to capture sequential information of users' shopping behavior. It extracts users' dynamic representations and scores user-item pair by calculating inner products between users' dynamic representations and items' embedding.

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fwqh7ulc4jj318t0cqgms.jpg)

The framework of DREAM:

1. Pooling operation on the items in a basket to get the representation of the basket. 
2. The input layer comprises a series of basket representations of a user. 
3. The dynamic representation of the user can be obtained in the hidden layer.
4. The output layer shows scores of this user towards all items.

References:

> Yu, Feng, et al. "A dynamic recurrent model for next basket recommendation." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2016.


## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
