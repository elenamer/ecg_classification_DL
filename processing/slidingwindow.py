'''

Label aggregation function taken from:
https://github.com/helme/ecg_ptbxl_benchmarking/blob/bed65591f0e530aa6a9cb4a4681feb49c397bf02/code/models/timeseries_utils.py#L534

'''

from processing.transform import Transform
import tensorflow as tf
import os
from evaluation.metrics import F1Metric
from wandb.keras import WandbCallback
import numpy as np


class SlidingWindow(Transform):
    def __init__(self, input_size):
        self.input_size = input_size
        self.idmap = [] 

    def reset_idmap(self):
        self.idmap = []

    def aggregate_labels(self, preds):
        '''
        needs to ba called right after process, meant to be used only in predict function
        '''
        aggregate_fn = np.mean
        if(self.idmap is not None and len(self.idmap)!=len(np.unique(self.idmap))):
            print("aggregating predictions...")
            preds_aggregated = []
            targs_aggregated = []
            for i in np.unique(self.idmap):
                preds_local = preds[np.where(self.idmap==i)[0]]
                preds_aggregated.append(aggregate_fn(preds_local,axis=0))
            return np.array(preds_aggregated)

    def process(self, X, labels=None, window=False):
        overlap = 0.5
        print("windowing")
        if window==True:
            new_data = []
            new_labels = []
            for ind, sig in enumerate(X):
                #print(sig)
                step = int(self.input_size*overlap)
                nrows = ((len(sig)-self.input_size)//step)+1
                print(nrows)
                windows = sig[step*np.arange(nrows)[:,None] + np.arange(self.input_size)]
                #print(windows)
                new_data.extend(windows.tolist())
                if labels is not None:
                    new_labels.extend([labels[ind]] * nrows)
                self.idmap.extend([ind] * nrows)
                #print(idmap)
        new_data = tf.keras.preprocessing.sequence.pad_sequences(new_data, maxlen=self.input_size, dtype="float32", padding="post", truncating="post", value=0.0)
        if new_data.ndim == 2:
            new_data = new_data[:,:,None]
        print(new_data.shape)
        if labels is None:
            return new_data
            # just crop/pad if needed
            # convert to numpy array

        new_labels = np.array(new_labels)
        print(new_labels.shape)
        return new_data, new_labels
