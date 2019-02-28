# Deep Learning for Next Basket Recommendation

This repository contains my implementations of [DREAM](http://www.nlpr.ia.ac.cn/english/irds/People/sw/DREAM.pdf) for next basket prediction.

## Requirements

- Python 3.6
- Pytorch 0.4 +
- Tensorflow 1.8 +
- Pandas 0.23 +
- scikit-learn 0.19 +
- Numpy
- Gensim

## Data

See data format in `data` folder which including the data sample files.

### Data Format

This repository can be used in other e-commerce datasets by two ways:
1. Modify your datasets into the same format of the sample.
2. Modify the data preprocess code in `data_helpers.py`.

Anyway, it should depends on what your data and task are.

## Network Structure

DREAM uses RNN to capture sequential information of users' shopping behavior. It extracts users' dynamic representations and scores user-item pair by calculating inner products between users' dynamic representations and items' embedding.

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fwqh7ulc4jj318t0cqgms.jpg)

The framework of DREAM:

1. Pooling operation on the items in a basket to get the representation of the basket. 
2. The input layer comprises a series of basket representations of a user. 
3. Dynamic representation of the user can be obtained in the hidden layer.
4. Finally the output layer shows scores of this user towards all items.

References:

> Yu, Feng, et al. "A dynamic recurrent model for next basket recommendation." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2016.


## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
