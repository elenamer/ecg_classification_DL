'''

Adapted from: 
https://github.com/ChihHanHuang/Deep-learning-ECG-The-China-Physiological-Signal-Challenge-2018-champion/blob/master/cpsc2018_champion/cpsc2018.py


'''


import glob
import random
import os
import argparse
import scipy.io as sio
from keras import backend as K
from sklearn.model_selection import train_test_split
import csv
import numpy
import numpy as np

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, LeakyReLU
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,  Input, Reshape
from tensorflow.keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D,concatenate,AveragePooling1D
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
import scipy.io as sio
from os import listdir

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs): 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform') 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer) 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint) 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint) 
            self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint) 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W) 
        if self.bias:
            uit += self.b 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u) 
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx()) 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

'''

set block parameters dynamically

'''

class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, dropout = 0, **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout

    def build(self, input_shape):
        self.conv1 = Convolution1D(12, 3, padding='same')
        self.lrelu1 = LeakyReLU(alpha=0.3)
        self.conv2 = Convolution1D(12, 3, padding='same')
        self.lrelu2 = LeakyReLU(alpha=0.3)
        self.conv3 = Convolution1D(12, 24, strides = 2, padding='same')
        self.lrelu3 = LeakyReLU(alpha=0.3)
        self.dp3 = Dropout(self.dropout)
        super().build(input_shape)

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)
        x = self.dp3(x)
        return x



class CPSCWinnerNet(tf.keras.layers.Layer):
    def __init__(self,dropout = 0, **kwargs):
        super(CPSCWinnerNet, self).__init__(**kwargs)
        self.num_blocks = 5
        self.dropout = dropout
        self.loss = 'categorical_crossentropy'

    def build(self, input_shape):
        self.conv1 = Convolution1D(12, 3, padding='same')
        self.lrelu1 = LeakyReLU(alpha=0.3)
        self.conv2 = Convolution1D(12, 3, padding='same')
        self.lrelu2 = LeakyReLU(alpha=0.3)
        self.conv3 = Convolution1D(12, 24, strides = 2, padding='same')
        self.lrelu3 = LeakyReLU(alpha=0.3)
        self.dp3 = Dropout(self.dropout)
        self.blocks = []
        for block in range(self.num_blocks-1):
            res_block = CNNBlock(self.dropout)
            self.blocks.append(res_block)
        self.gru = Bidirectional(GRU(12, input_shape=(2250,12),return_sequences=True,return_state=False))
        self.lrelu4 = LeakyReLU(alpha=0.3)
        self.dp4 = Dropout(self.dropout)
        self.att = AttentionWithContext()
        self.bn = BatchNormalization()
        self.lrelu5 = LeakyReLU(alpha=0.3)
        self.dp5 = Dropout(self.dropout)
        super().build(input_shape)

    def get_optimizer(self, lr):
        return tf.keras.optimizers.Adam(lr=lr)

    def call(self, x, **kwargs):
        x = self.conv1(x) 
        x = self.lrelu1(x) 
        x = self.conv2(x) 
        x = self.lrelu2(x) 
        x = self.conv3(x) 
        x = self.lrelu3(x) 
        x = self.dp3(x) 
        for block in self.blocks:
            x = block(x)
        x = self.gru(x) 
        x = self.lrelu4(x) 
        x = self.dp4(x) 
        x = self.att(x) 
        x = self.bn(x) 
        x = self.lrelu5(x) 
        x = self.dp5(x) 
        return x
