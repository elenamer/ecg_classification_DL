#Something like transforms, have it as an argument in dataset and call it

from processing.transform import Transform
import tensorflow as tf
import os
import numpy as np


class RepeatCrop(Transform):
    def __init__(self, input_size):
        self.name = "repeatcrop"
        self.input_size = input_size
    # Add something like normalization by default

    def reset_idmap(self):
        self.idmap = []

    def aggregate_labels(self, preds):
        return preds


    def process(self, X, labels=None):
        n = self.input_size
        full_data = []
        full_labels = []
        for sig in X:
            data_list = np.zeros((n), dtype = np.float32)
            a = sig.shape[0]

            if a > n:
                ran_b = np.random.randint(int(n/2),a-int(n/2))
                data_list = sig[ran_b-int(n/2):ran_b+int(n/2)] 
            
            if a == n:
                data_list = sig
            
            if int(n/2) <= a < n:
                data_list[0:a] = sig
                data_list[a:n] = sig[0:n-a]
                
            if int(n/3) <= a < int(n/2):
                data_list[0:a] = sig
                data_list[a:2*a] = sig
                data_list[2*a:n] = sig[0:n-2*a]        

            if  a < int(n/3):
                data_list[0:a] = sig
                data_list[a:2*a] = sig
                data_list[2*a:3*a] = sig
                data_list[3*a:n] = sig[0:n-3*a]  
            full_data.append(data_list)

        new_data = np.array(full_data)

        if new_data.ndim == 2:
            new_data = new_data[:,:,None]
        print(new_data.shape)
        if labels is None:
            return new_data
        new_labels = np.array(labels)
        print(new_labels.shape)
        return new_data, new_labels, self.idmap


    # def call(input signals/patient ids):
    #     maybe cropping
    #     maybe padding
    #     maaaaybe do label encoding here?
    #     return segmented beats/windows
