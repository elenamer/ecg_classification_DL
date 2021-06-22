# !!!!

from datasets.dataset import Dataset
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


results_path = "./data_overviews"


initial_classes_dict = {
    "aami" : {
        "NONE" : 0, # N
        "LBBB" : 1, # N 
        "LBBBB" : 1, # N
        "LFBBB" : 1, # N
        "RBBB" : 1, # N
        "PAC" : 2, # S
        "JPT" : 2, # S
        "VEB" : 3, # V
        "VPB" : 3 # V
    },
    'rhythm' : {
        'SB' :0,
        'SR':1,  
        'AFIB':2, 
        'ST':3,   
        'AF':4,  
        'SI':5,  
        'SVT':6, 
        'AT':7,   
        'AVNRT':8,
        'AVRT':9,
        'SAAW':10,
        'SA':11
    }

}

class Arr10000Dataset(Dataset):

    def __init__(self): ## classes, segmentation, selected channel
        self.name = 'arr10000'
        super(Arr10000Dataset,self).__init__()
        
        self.path = "./data/"+self.name


        self.index = []
        self.patientids = self.get_patientids()

        self.lead = "II"
        self.freq = 500

        self.encoded_labels = self.encode_labels()


        #self.num_channels = n_channels
        #with open(self.path+"RECORDS"+choice.upper()) as f:
        #    self.patientids = f.read().splitlines()
        #    #patientids = [os.path.split(id)[-1] for id in patientids]		
        #self.classes = "rhythmic"#classes
        #self.classes_dict = initial_classes_dict[self.classes]


    def get_patientids(self):
        Y = pd.read_excel(self.path+os.sep+"Diagnostics.xlsx", index_col=None, header=0, engine='openpyxl') 
        self.index = Y
        self.index .set_index("FileName", inplace=True, drop=False)
        return Y.FileName.values.tolist()

    def get_signal(self, path, idx):
        # to implement
        X = pd.read_csv(path+os.sep+idx, header=0)
        #plt.plot(X[[self.lead]].values)
        #plt.show()
        return X[[self.lead]].values

    def get_annotation(self, path, idx):
        Y = self.index
        row=Y[Y.index==idx]
        #print(row)
        beats = str(row.Beat.values[0]).split(sep=" ")
        #print(beats)
        #print(self.morphological_classes.keys())
        labls = [self.morphological_classes[l] for l in beats if l in self.morphological_classes.keys()]
        rhytms = [self.rhythmic_classes[row.Rhythm.values[0]] if row.Rhythm.values[0] in self.rhythmic_classes.keys() else None]
        return labls, rhytms

    def examine_database(self):

        # load and convert annotation data
        Y = self.index
        #print(Y.Rhythm) 
        #print(Y.scp_codes)
        # Load raw signal data
        #X = self.load_raw_data(Y, sampling_rate)

        unique_rhythm = Counter(Y.Rhythm)

        count=0
        beat_labels = []
        for ind, row in Y.iterrows():
            #if self.classes == "aami":
            beats = str(row.Beat).split(sep=" ")
            labls = [l for l in beats if l in self.classes_dict.keys()]
            if len(labls)>1:
                count+=1
                print(labls)
            beat_labels+=labls
        print(count)
        unique_beats = Counter(beat_labels)
        if self.classes == "rhythm":
            print("in rhythm")
            unique_beats.update(unique_rhythm)
        results_df = pd.DataFrame.from_dict({"all":unique_beats}, orient='index')
        results_df.to_csv(results_path+os.sep+self.name+"_distribution_"+self.classes+".csv")
