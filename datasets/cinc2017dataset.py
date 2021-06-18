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

choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

WFDB = "/usr/local/bin"#/home/elena/wfdb/bin"

#sampling_rate=200

class CincChallenge2017Dataset(Dataset):

    def __init__(self): ## classes, segmentation, selected channel
        self.name = 'cinc2017'#name
        super(CincChallenge2017Dataset, self).__init__()
        self.path = "./data/"+self.name
        self.patientids = self.get_recordids()
        self.index = self.get_index()
  
        print(self.patientids)
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
        Reads .dat file and returns in numpy array. Assumes 2 channels.  The
        returned array is n x 3 where n is the number of samples. The first column
        is the sample number and the second two are the first and second channel
        respectively.
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
        rdsamp = os.path.join(WFDB, 'rdsamp')
        output = subprocess.check_output([rdsamp, '-r', idx], cwd=path+os.sep+"training2017")
        data = np.fromstring(output, dtype=np.int32, sep=' ')
        return data.reshape((-1, 2))[:,1]

    def get_annotation(self, path, idx):
        row = self.index[self.index.ID==idx]
        beats = [row.Label.values[0]]
        labls = [self.morphological_classes[str(l)] for l in beats if str(l) in self.morphological_classes.keys()]
        rhytms = [self.rhythmic_classes[str(l)] for l in beats if str(l) in self.rhythmic_classes.keys()]
        return labls, rhytms
            

    # def get_class_distributions(self):
        mydict_labels = {}
        mydict_rhythms = {}
        
        labels=[]
        rhythms=[]

        # load and convert annotation data
        Y = pd.read_csv(self.path+os.sep+'REFERENCE-v3.csv', index_col=None, header=None, names=["record","class_label"])
        labels = Y.class_label
        print(self.morphological_classes.keys())
        rhythms = [l for l in list(labels) if l in self.rhythmic_classes.keys()] 
        if len(self.morphological_classes.keys()) > 0:
            labels = [l for l in list(labels) if l in self.morphological_classes.keys()] 
            mydict_labels = Counter(labels)
            results_df_lab = pd.DataFrame.from_dict({"all":mydict_labels}, orient='index')
        else:
            morph_class_ids = list(set(self.morphological_classes.values()))
            results_df_lab = pd.DataFrame(np.zeros((1, len(morph_class_ids))),columns=morph_class_ids)
            results_df_lab.loc['all',:] = results_df_lab.sum()

        if len(self.rhythmic_classes.keys()) > 0:
            rhythms = [l for l in list(labels) if l in self.rhythmic_classes.keys()] 
            mydict_rhythms = Counter(rhythms)
            results_df_rhy = pd.DataFrame.from_dict({"all":mydict_rhythms}, orient='index')

        beat_label_counts = np.ones(len(mydict_rhythms.keys()))
        rhythm_label_counts = []

        print(rhythms)
        #print(labels)
        return results_df_lab.loc['all',:], results_df_rhy.loc['all',:]