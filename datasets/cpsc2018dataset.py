from datasets.dataset import Dataset
import os
import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter
import subprocess
import ast
from scipy.io import loadmat
import matplotlib.pyplot as plt
from skmultilearn.model_selection import iterative_train_test_split
import itertools


choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"
leads = [ 'I','II', 'III', 'aVL','aVF', 'V1','V2','V3','V4','V5','V6']

#sampling_rate=200

class CPSC2018Dataset(Dataset):

    def __init__(self, task, lead='II'): ## classes, segmentation, selected channel
        self.name = 'cpsc2018'#name
        super(CPSC2018Dataset, self).__init__(task)
        self.path = "./data/"+self.name
        #with open(self.path+"RECORDS"+choice.upper()) as f:


        self.lead = lead
        self.freq = 500
        self.index = []

        self.patientids = self.get_recordids()
        max_size=len(self.patientids) # FOr now
        # Load PTB-XL data
        self.data = [self.get_signal(self.path,id) for id in self.index.index[:max_size]]
        
        self.encoded_labels = self.encode_labels()

    def get_recordids(self):
        Y = pd.read_csv(self.path+os.sep+'REFERENCE.csv', index_col=None, header=0)
        self.index = Y
        self.index.set_index("Recording",inplace=True, drop=False)
        return Y.Recording.values.tolist()

    def get_signal(self, path, idx):
        # implement ecg signal extraction
        sex, age, sig = loadmat(path+os.sep+"TrainingSet"+os.sep+idx+".mat")['ECG'][0][0] 
        lead_names = self.lead.split("-")
        if len(lead_names) == 1:
            signal = sig.T[:,leads.index(lead_names[0])]
        else:
            signal = sig.T[:,leads.index(lead_names[0])] - sig.T[:,leads.index(lead_names[1])]
        # plt.plot(signal)
        # plt.show()
        # plt.plot(sig.T[:,leads.index('II')])
        # plt.show()
        return signal

    def get_annotation(self, path, idx):
        row=self.index[self.index.index==idx]
        #print(row)
        beats = [row.First_label.values[0], row.Second_label.values[0], row.Third_label.values[0]]
        labls = [self.classes[str(int(l))] for l in beats if not np.isnan(l) and str(int(l)) in self.classes.keys()]
        rhytms = [self.classes[str(int(l))] for l in beats if not np.isnan(l) and str(int(l)) in self.classes.keys()]
        labls.extend(rhytms)
        return labls

    # def get_index(self):
    #     self.index = Y
    #     return Y

    def examine_database(self):
        #print(self.patientids)
        mydict_labels = {}
        mydict_rhythms = {}
        labels=[]
        rhythms=[]

        # load and convert annotation data
        Y = pd.read_csv(self.path+os.sep+'REFERENCE.csv', index_col=None, header=0)
        print(Y)
        unique_rhythm = Counter(pd.concat([Y.First_label, Y.Second_label.dropna(), Y.Third_label.dropna()]))
        results_df = pd.DataFrame.from_dict({"all":unique_rhythm}, orient='index')
        results_df.to_csv(results_path+os.sep+self.name+"_distribution.csv")
    

    def get_crossval_splits(self, split=9):
        max_size=len(self.patientids)
        data=np.array(self.data, dtype=object)
        temp_labels = self.encoded_labels.iloc[:max_size,:]
        
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
        mask = np.ones(data.shape[0],dtype=bool) # keep only train indices to one
        mask[test]=0
        mask[val]=0
        X_train, y_train = data[mask], labels[mask]
 
        print(X_test.shape)
        print(X_val.shape)

        # Preprocess signal data
        #self.X_train, self.X_val, self.X_test = preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        # self.n_classes = self.y_train.shape[1]
        # partition = {"train": self.y_test.filename_lr.values.tolist() ,"validation":self.y_test.filename_lr.values.tolist(), "test":self.y_test.filename_lr.values.tolist()}
        print(y_test.shape)
        print(y_train.shape)
        print(y_val.shape)
        return X_train, y_train, X_val, y_val, X_test, y_test


