import os
import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter
import subprocess
import wfdb
import ast

choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

#sampling_rate=200

class CincChallenge2017Dataset():

    def __init__(self): ## classes, segmentation, selected channel
        self.name = 'cinc-challenge-2017'#name
        self.path = "./data/"+self.name+"/training"
        self.patientids = self.get_recordids()
        print(self.patientids)
        #patientids = [os.path.split(id)[-1] for id in patientids]		

    def get_recordids(self):
        with open(self.path+os.sep+"RECORDS"+choice.upper()) as f:
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

        # load and convert annotation data
        Y = pd.read_csv(self.path+os.sep+'REFERENCE-v3.csv', index_col=None, header=None, names=["record","class_label"])
        print(Y)
        unique_rhythm = Counter(Y.class_label)
        results_df = pd.DataFrame.from_dict({"all":unique_rhythm}, orient='index')
        results_df.to_csv(results_path+os.sep+self.name+"_distribution.csv")
