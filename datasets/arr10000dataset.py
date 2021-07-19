# !!!!

from datasets.dataset import Dataset
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from skmultilearn.model_selection import iterative_train_test_split
import itertools

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

    def __init__(self, task, lead = 'II'): ## classes, segmentation, selected channel
        self.name = 'arr10000'
        super(Arr10000Dataset,self).__init__(task)
        
        self.path = "./data/"+self.name


        self.index = []
        self.patientids = self.get_patientids()

        self.lead = lead # options: e.g. 'II' or 'V1-V2'
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
        lead_names = self.lead.split("-")
        if len(lead_names) == 1:
            sig = X[[self.lead]].values
        else:
            sig = X[[lead_names[0]]].values - X[[lead_names[1]]].values
        # plt.plot(sig)
        # plt.show()
        # plt.plot(X[['II']].values)
        # plt.show()
        return sig


    def get_annotation(self, path, idx):
        Y = self.index
        row=Y[Y.index==idx]
        #print(row)
        beats = str(row.Beat.values[0]).split(sep=" ")
        #print(beats)
        #print(self.morphological_classes.keys())
        labls = [self.classes[l] for l in beats if l in self.classes.keys()]
        rhytms = [self.classes[row.Rhythm.values[0]] if row.Rhythm.values[0] in self.classes.keys() else None]
        labls.extend(rhytms)
        return labls

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


    def get_crossval_splits(self, split=9):
        max_size=2200 # FOr now
        # Load PTB-XL data
        data = [self.get_signal(self.path+os.sep+"ECGData",id+".csv") for id in self.index.index[:max_size]]
        
        data=np.array(data, dtype=object)
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
