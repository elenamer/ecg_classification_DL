#x:(n_samples, beat_length)
#y:(n_samples)

import os
import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter
import subprocess
import wfdb
from .utils import *

choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

WFDB = "/usr/local/bin"#/home/elena/wfdb/bin"

sampling_rate=100

# class translation
# main classes
# rare classes

# freq.
# segment length
# multilabel (yes, no)
# 

initial_classes_dict = {
    "form" : {
    },
    "aami" : {
        "NORM" : 0, # N
        "CLBBB" : 1, # N 
        "ILBBB" : 1, # N
        "CRBBB" : 1, # N
        "IRBBB" : 1, # N
        "PAC" : 2, # S
        "SVARR" : 2, # S
        "PVC" : 3, # V
        "PRC(S)" : 3 # S or V
    },
    'rhythm' : {
        'AFIB':0,
        'AFLT':1,
        'SR' :2,
        '2AVB':3,
    }

}

class PTBXLDataset():

    def __init__(self, classes): ## classes, segmentation, selected channel
        self.name = 'ptb-xl'
        self.path = "./data/"+self.name+"/"
        self.num_channels = 12
        self.patientids = self.get_patientids()
        self.classes=classes
        self.classes_dict = initial_classes_dict[classes]
        #print(self.patientids)
            #patientids = [os.path.split(id)[-1] for id in patientids]		

    def get_patientids(self):
        with open(self.path+"RECORDS"+choice.upper()) as f:
            return f.read().splitlines()

    def load_raw_data(self, df, sampling_rate):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def extract_wave(self, idx):
        """
        Reads .dat file and returns in numpy array. Assumes 2 channels.  The
        returned array is n x 3 where n is the number of samples. The first column
        is the sample number and the second two are the first and second channel
        respectively.
        """
        rdsamp = os.path.join(WFDB, 'rdsamp')
        output = subprocess.check_output([rdsamp, '-r', idx], cwd=self.path)
        data = np.fromstring(output, dtype=np.int32, sep=' ')
        return data.reshape((-1, self.num_channels+1))
    
    def examine_database(self):
        #print(self.patientids)
        mydict_labels = {}
        mydict_rhythms = {}
        labels=[]
        rhythms=[]

        self.data, self.raw_labels = load_dataset(self.path, sampling_rate)
        # Preprocess label data
        self.labels = compute_label_aggregations(self.raw_labels, self.path, self.classes)
        self.data, self.labels, self.Y, _ = select_data(self.data, self.labels, self.classes, 0, self.path+'exprs/data/')


        # load and convert annotation data
        if self.classes == "rhythm" or self.classes == "form":
            Y = self.labels #pd.read_csv(self.path+os.sep+'ptbxl_database.csv', index_col='ecg_id')
        else:
            pd.read_csv(self.path+os.sep+'ptbxl_database.csv', index_col='ecg_id')
        #Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        #print(Y.scp_codes)
        # Load raw signal data
        #X = self.load_raw_data(Y, sampling_rate)


        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(self.path+os.sep+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]


        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # Apply diagnostic superclass
        #Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
        #print(Y.scp_codes)
        count=0
        for ind, row in Y.iterrows():
            if self.classes == 'aami':  
                labls = [l for l in row.scp_codes.keys() if l in self.classes_dict.keys()] 
            elif self.classes == 'form':
                labls = row.form
            elif self.classes == 'rhythm':
                labls = row.rhythm
            #print(labls)
            if len(labls)>1:
                count+=1
                print(labls)
            labels+=list(labls)
        print(count)
        #print(labels)
        unique_rhythm = Counter(labels)
        results_df = pd.DataFrame.from_dict({"all":unique_rhythm}, orient='index')
        results_df.to_csv(results_path+os.sep+"ptbxl_distribution_"+self.classes+".csv")

    def get_crossval_split(self, id=9):
        # Load PTB-XL data
        self.data, self.raw_labels = load_dataset(self.path, sampling_rate)
        # Preprocess label data
        self.labels = compute_label_aggregations(self.raw_labels, self.path, self.classes)
        self.data, self.labels, self.Y, _ = select_data(self.data, self.labels, self.classes, 0, self.path+'exprs/data/')

        



        # 10th fold for testing (9th for now)
        #self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.labels[self.labels.strat_fold == id]
        # 9th fold for validation (8th for now)
        #self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.labels[self.labels.strat_fold == id-1]
        # rest for training
        #self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.labels[self.labels.strat_fold <= id-2]

        # Preprocess signal data
        #self.X_train, self.X_val, self.X_test = preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        self.n_classes = self.y_train.shape[1]
        partition = {"train": self.y_test.filename_lr.values.tolist() ,"validation":self.y_test.filename_lr.values.tolist(), "test":self.y_test.filename_lr.values.tolist()}
        return partition

    def get_labels(self):
        print(self.labels)
        #labls = 
        return self.labels

# Split data into train and test
# test_fold = 10
# Train
# X_train = X[np.where(Y.strat_fold != test_fold)]
# y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
# X_test = X[np.where(Y.strat_fold == test_fold)]
# y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass