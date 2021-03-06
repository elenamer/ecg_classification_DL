'''

Label aggregation function taken from:
https://github.com/helme/ecg_ptbxl_benchmarking/blob/bed65591f0e530aa6a9cb4a4681feb49c397bf02/code/models/timeseries_utils.py#L534

'''


#Something like transforms, have it as an argument in dataset and call it

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

class Transform():

    def __init__(self, input_size):
        self.name = "standard"
        self.input_size = input_size
        self.idmap = []
    # Add something like normalization by default

    def reset_idmap(self):
        self.idmap = []

    def aggregate_labels(self, preds, idmap=None):
        '''
        needs to be called right after process, meant to be used only in predict function
        '''
        aggregate_fn = np.mean
        print(idmap)
        if idmap is not None:
            print("aggregating predictions...")
            preds_aggregated = []
            targs_aggregated = []
            for i in np.unique(idmap):
                preds_local = preds[np.where(idmap==i)[0]]
                preds_aggregated.append(aggregate_fn(preds_local,axis=0))
            return np.array(preds_aggregated)
        else:
            return np.array(preds)

    def process(self, X, labels=None):
        new_data = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.input_size, dtype="float32", padding="post", truncating="post", value=0.0)
        if new_data.ndim == 2:
            new_data = new_data[:,:,None]
        print(new_data.shape)
        if labels is None:
            return new_data
        new_labels = np.array(labels)
        print(new_labels.shape)
        return new_data, new_labels, self.idmap