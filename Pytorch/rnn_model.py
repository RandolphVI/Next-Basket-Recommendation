# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import torch
from torch.autograd import Variable
from utils import data_helpers as dh


class DRModel(torch.nn.Module):
    """
    Input Data: b_1, ... b_i ..., b_t
                b_i stands for user u's ith basket
                b_i = [p_1,..p_j...,p_n]
                p_j stands for the  jth product in user u's ith basket
    """

    def __init__(self, config):
        super(DRModel, self).__init__()

        # Model configuration
        self.config = config

        # Layer definitions
        # Item embedding layer, item's index
        self.encode = torch.nn.Embedding(num_embeddings=config.num_product,
                                         embedding_dim=config.embedding_dim,
                                         padding_idx=0)
        self.pool = {'avg': dh.pool_avg, 'max': dh.pool_max}[config.basket_pool_type]  # Pooling of basket

        # RNN type specify
        if config.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(torch.nn, config.rnn_type)(input_size=config.embedding_dim,
                                                          hidden_size=config.embedding_dim,
                                                          num_layers=config.rnn_layer_num,
                                                          batch_first=True,
                                                          dropout=config.dropout,
                                                          bidirectional=False)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[config.rnn_type]
            self.rnn = torch.nn.RNN(input_size=config.embedding_dim,
                                    hidden_size=config.embedding_dim,
                                    num_layers=config.rnn_layer_num,
                                    nonlinearity=nonlinearity,
                                    batch_first=True,
                                    dropout=config.dropout,
                                    bidirectional=False)

    def forward(self, x, lengths, hidden):
        # Basket Encoding
        # users' basket sequence
        ub_seqs = torch.Tensor(self.config.batch_size, self.config.seq_len, self.config.embedding_dim)
        for (i, user) in enumerate(x):  # shape of x: [batch_size, seq_len, indices of product]
            embed_baskets = torch.Tensor(self.config.seq_len, self.config.embedding_dim)
            for (j, basket) in enumerate(user):  # shape of user: [seq_len, indices of product]
                basket = torch.LongTensor(basket).resize_(1, len(basket))
                basket = self.encode(torch.autograd.Variable(basket))  # shape: [1, len(basket), embedding_dim]
                basket = self.pool(basket, dim=1)
                basket = basket.reshape(self.config.embedding_dim)
                embed_baskets[j] = basket  # shape:  [seq_len, 1, embedding_dim]
            # Concat current user's all baskets and append it to users' basket sequence
            ub_seqs[i] = embed_baskets  # shape: [batch_size, seq_len, embedding_dim]

        # Packed sequence as required by pytorch
        packed_ub_seqs = torch.nn.utils.rnn.pack_padded_sequence(ub_seqs, lengths, batch_first=True)

        # RNN
        output, h_u = self.rnn(packed_ub_seqs, hidden)

        # shape: [batch_size, true_len(before padding), embedding_dim]
        dynamic_user, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return dynamic_user, h_u

    def init_weight(self):
        # Init item embedding
        initrange = 0.1
        self.encode.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # Init hidden states for rnn
        weight = next(self.parameters()).data
        if self.config.rnn_type == 'LSTM':
            return (Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_()),
                    Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_()))
        else:
            return Variable(torch.zeros(self.config.rnn_layer_num, batch_size, self.config.embedding_dim))
