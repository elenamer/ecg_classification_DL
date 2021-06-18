# !!!!

from datasets.dataset import Dataset
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


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

        self.encode_labels()


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

    def encode_labels(self):
        mlb_morph = MultiLabelBinarizer(classes=range(len(self.all_morph_classes.values)))
        mlb_rhy = MultiLabelBinarizer(classes=range(len(self.all_rhy_classes.values)))
        encoded_index = self.index.copy()
        #encoded_index.set_index("FileName", inplace=True)
        print(encoded_index)
        encoded_index["rhythms_mlb"] = ""
        encoded_index["beats_mlb"] = ""
        for ind in encoded_index.index:
            #print(row)
            labls, rhythms = self.get_annotation(self.path, ind)
            encoded_index.at[ind, "beats_mlb"] = tuple(labls)
            encoded_index.at[ind, "rhythms_mlb"] = tuple(rhythms)

        encoded_index["beats_mlb"] = mlb_morph.fit_transform(encoded_index["beats_mlb"]).tolist()
        print(encoded_index["beats_mlb"])
        encoded_index["rhythms_mlb"] = mlb_rhy.fit_transform(encoded_index["rhythms_mlb"]).tolist()
        print(encoded_index["rhythms_mlb"])
        encoded_index = encoded_index[["beats_mlb","rhythms_mlb"]]
        print(encoded_index)

        
    def load_raw_data(self, df, sampling_rate):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

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
