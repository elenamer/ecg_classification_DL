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

class CPSC2018Dataset():

    def __init__(self): ## classes, segmentation, selected channel
        self.name = 'cpsc-2018'#name
        self.path = "./data/"+self.name
        #with open(self.path+"RECORDS"+choice.upper()) as f:
        self.patientids = self.get_recordids()
        print(self.patientids)
        #    #patientids = [os.path.split(id)[-1] for id in patientids]		

    def get_recordids(self):
        Y = pd.read_csv(self.path+os.sep+'REFERENCE.csv', index_col=None, header=0)
        return Y.Recording.values.tolist()

    def extract_wave(self, idx):
        # implement ecg signal extraction
        return data
    
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
