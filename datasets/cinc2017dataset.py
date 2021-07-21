from datasets.dataset import Dataset
import os
import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter
import subprocess
import wfdb
import ast
import matplotlib.pyplot as plt
from skmultilearn.model_selection import iterative_train_test_split
import itertools


choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

WFDB = "/usr/local/bin"#/home/elena/wfdb/bin"

#sampling_rate=200

class CincChallenge2017Dataset(Dataset):

    def __init__(self, task): ## classes, segmentation, selected channel
        self.name = 'cinc2017'#name
        super(CincChallenge2017Dataset, self).__init__(task)
        self.path = "./data/"+self.name
        self.patientids = self.get_recordids()
        self.index = self.get_index()
        self.freq = 300
        print(self.patientids)

        self.encoded_labels = self.encode_labels()
        #patientids = [os.path.split(id)[-1] for id in patientids]		

    def get_recordids(self):
        with open(self.path+os.sep+"/training2017"+os.sep+"RECORDS"+choice.upper()) as f:
            patientids = f.read().splitlines()
        return patientids

    def load_raw_data(self, df, sampling_rate):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def extract_wave(self, idx):
        """
        Deprecated 
        """
        rdsamp = os.path.join(WFDB, 'rdsamp')
        output = subprocess.check_output([rdsamp, '-r', idx], cwd=self.path+os.sep+"training2017")
        data = np.fromstring(output, dtype=np.int32, sep=' ')
        return data.reshape((-1, self.num_channels+1))
    
    def examine_database(self):
        #print(self.patientids)
        mydict_labels = {}
        mydict_rhythms = {}
        labels=[]
        rhythms=[]

        # load and convert annotation data
        Y = pd.read_csv(self.path+os.sep+'REFERENCE-v3.csv', index_col=None, header=None, names=["record","class_label"])
        print(Y)
        unique_rhythm = Counter(Y.class_label)
        results_df = pd.DataFrame.from_dict({"all":unique_rhythm}, orient='index')
        results_df.to_csv(results_path+os.sep+self.name+"_distribution.csv")

    def get_index(self):
        Y = pd.read_csv(self.path+os.sep+'REFERENCE-v3.csv', index_col=None, header=None)
        Y.columns = ["ID","Label"]
        Y.set_index("ID", inplace=True, drop=False)
        return Y


    def get_signal(self, path, idx):
        data, metadata = wfdb.rdsamp(path+os.sep+idx)
        return data[:,0]


    def get_annotation(self, path, idx):
        row = self.index[self.index.ID==idx]
        beats = [row.Label.values[0]]
        labls = [self.classes[str(l)] for l in beats if str(l) in self.classes.keys()]
        rhytms = [self.classes[str(l)] for l in beats if str(l) in self.classes.keys()]
        labls.extend(rhytms)
        return labls
            

    def get_crossval_splits(self, split=9):
        max_size=2200 # FOr now
        # Load PTB-XL data
        data = [self.get_signal(self.path+os.sep+'training2017',id) for id in self.index.index[:max_size]]
        
        temp_labels = self.encoded_labels.iloc[:max_size,:]
        data=np.array(data, dtype=object)
        print("before")
        # Preprocess label data

        print(temp_labels.loc[:,"labels_mlb"].values.shape)

        data = data[(temp_labels.labels_mlb.apply(lambda x:sum(x)) > 0 ).values]
        labels = np.array(temp_labels.loc[(temp_labels.labels_mlb.apply(lambda x:sum(x)) > 0 ).values,"labels_mlb"].values.tolist())

        train, test= next(itertools.islice(self.k_fold.split(data,labels), split, None))
        X_test, y_test = data[test], labels[test]
        if split != 0:
            val_split = split - 1
        else:
            val_split = self.k_fold.n_splits - 1
        # for now always have a different validation set
        train, val= next(itertools.islice(self.k_fold.split(data,labels), val_split, None))
        X_val, y_val = data[val], labels[val]
        mask = np.ones(data.shape,dtype=bool) # keep only train indices to one
        mask[test]=0
        mask[val]=0
        X_train, y_train = data[mask], labels[mask]

        print(X_test.shape)
        print(X_val.shape)

        print(y_test.shape)
        print(y_train.shape)
        print(y_val.shape)
        return X_train, y_train, X_val, y_val, X_test, y_test
