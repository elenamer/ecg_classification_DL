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

choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

#sampling_rate=200

class CPSC2018Dataset(Dataset):

    def __init__(self): ## classes, segmentation, selected channel
        self.name = 'cpsc2018'#name
        super(CPSC2018Dataset, self).__init__()
        self.path = "./data/"+self.name
        #with open(self.path+"RECORDS"+choice.upper()) as f:


        self.lead = "II"
        self.lead_id = 1 # temp, this is actually determined in extract_metadata

        self.index = []

        self.patientids = self.get_recordids()

        self.encoded_labels = self.encode_labels()

    def get_recordids(self):
        Y = pd.read_csv(self.path+os.sep+'REFERENCE.csv', index_col=None, header=0)
        self.index = Y
        self.index.set_index("Recording",inplace=True, drop=False)
        return Y.Recording.values.tolist()

    def get_signal(self, path, idx):
        # implement ecg signal extraction
        sex, age, sig = loadmat(path+os.sep+"TrainingSet"+os.sep+idx+".mat")['ECG'][0][0] 
        plt.plot(sig.T[:,self.lead_id])  
        plt.show()
        return sig.T[:,self.lead_id]

    def get_annotation(self, path, idx):
        row=self.index[self.index.index==idx]
        #print(row)
        beats = [row.First_label.values[0], row.Second_label.values[0], row.Third_label.values[0]]
        labls = [self.morphological_classes[str(int(l))] for l in beats if not np.isnan(l) and str(int(l)) in self.morphological_classes.keys()]
        rhytms = [self.rhythmic_classes[str(int(l))] for l in beats if not np.isnan(l) and str(int(l)) in self.rhythmic_classes.keys()]
        return labls, rhytms

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
    
    # def get_class_distributions(self):
    #     mydict_labels = {}
    #     mydict_rhythms = {}

    #     labels=[]
    #     rhythms=[]

    #     beat_label_counts = []
    #     rhythm_label_counts = []

    #     # load and convert annotation data
    #     Y = pd.read_csv(self.path+os.sep+'REFERENCE.csv', index_col=None, header=0, dtype=str)
    #     labels = pd.concat([Y.First_label, Y.Second_label.dropna(), Y.Third_label.dropna()]).values
    #     print(self.morphological_classes.keys())
    #     rhythms = [l for l in list(labels) if l in self.rhythmic_classes.keys()] 
    #     labels = [l for l in list(labels) if l in self.morphological_classes.keys()] 
    #     print(rhythms)
    #     print(labels)

    #     mydict_rhythms = Counter(rhythms)
    #     mydict_labels = Counter(labels)

    #     results_df_rhy = pd.DataFrame.from_dict({"all":mydict_rhythms}, orient='index')
    #     results_df_lab = pd.DataFrame.from_dict({"all":mydict_labels}, orient='index')

    #     return results_df_lab.loc['all',:], results_df_rhy.loc['all',:]

