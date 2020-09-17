# Deep Learning for Next Basket Recommendation

This repository contains my implementations of [DREAM](http://www.nlpr.ia.ac.cn/english/irds/People/sw/DREAM.pdf) for next basket prediction.

## Requirements

- Python 3.6
- Pytorch 1.6.0
- Pandas 1.1.2
- Scikit-learn 0.19.1
- Numpy 1.16.2
- Gensim 3.5.0
- Tqdm 4.49.0 

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

![](https://live.staticflickr.com/65535/49612979743_33d836d5a4_o.png)

The framework of DREAM:

1. Pooling operation on the items in a basket to get the representation of the basket. 
2. The input layer comprises a series of basket representations of a user. 
3. The dynamic representation of the user can be obtained in the hidden layer.
4. The output layer shows scores of this user towards all items.

References:

> Yu, Feng, et al. "A dynamic recurrent model for next basket recommendation." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2016.


## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Ph.D.

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
