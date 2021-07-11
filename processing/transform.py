#Something like transforms, have it as an argument in dataset and call it

import tensorflow as tf
import os
import numpy as np


class Transform():

    def __init__(self, input_size):
        self.name = "standard"
        self.input_size = input_size
    # Add something like normalization by default

    def aggregate_labels(self, preds, idmap):
        print("are you here?")
        raise NotImplementedError


    def process(self, X, labels=None):
        new_data = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.input_size, dtype="float32", padding="post", truncating="post", value=0.0)
        if new_data.ndim == 2:
            new_data = new_data[:,:,None]
        print(new_data.shape)
        if labels is None:
            return new_data
        new_labels = np.array(labels)
        print(new_labels.shape)
        return new_data, new_labels


    # def call(input signals/patient ids):
    #     maybe cropping
    #     maybe padding
    #     maaaaybe do label encoding here?
    #     return segmented beats/windows