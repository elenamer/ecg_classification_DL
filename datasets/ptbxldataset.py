#x:(n_samples, beat_length)
#y:(n_samples)

from datasets.dataset import Dataset
import os
import numpy as np
import pandas as pd
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

class PTBXLDataset(Dataset):

    def __init__(self): ## classes, segmentation, selected channel
        self.name = 'ptb-xl'
        super(PTBXLDataset, self).__init__()
        self.path = "./data/"+self.name+"/"
        self.num_channels = 12
        self.patientids = self.get_patientids()
        self.classes='rhythm'

        self.freq = 100
        self.lead = "II"
        self.lead_id = 1 # temp, this is actually determined in extract_metadata

        self.index = self.get_index()

        #self.classes_dict = initial_classes_dict[classes]
        #print(self.patientids)
            #patientids = [os.path.split(id)[-1] for id in patientids]		

    def get_patientids(self):
        with open(self.path+"RECORDS"+choice.upper()) as f:
            return f.read().splitlines()

    def get_index(self):
        #print(self.patientids)

        self.data, self.raw_labels = load_dataset(self.path, sampling_rate)
        # Preprocess label data
        ## modify this function 

        self.labels = compute_label_aggregations(self.raw_labels, self.path, self.classes)
        self.data, self.labels, self.Y, _ = select_data(self.data, self.labels, self.classes, 0, self.path+'exprs/data/')


        # # load and convert annotation data
        # if self.classes == "rhythm" or self.classes == "form":
        Y = self.labels #pd.read_csv(self.path+os.sep+'ptbxl_database.csv', index_col='ecg_id')
        Y.set_index("filename_lr", inplace=True, drop=False)
        return Y


    def load_raw_data(self, df, sampling_rate):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(self.path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def get_signal(self, path, idx):
        """
        Reads .dat file and returns in numpy array. Assumes 2 channels.  The
        returned array is n x 3 where n is the number of samples. The first column
        is the sample number and the second two are the first and second channel
        respectively.
        """
        rdsamp = os.path.join(WFDB, 'rdsamp')
        output = subprocess.check_output([rdsamp, '-r', idx], cwd=path)
        data = np.fromstring(output, dtype=np.int32, sep=' ')
        # plt.plot(data.reshape((-1, self.num_channels+1))[:,self.lead_id])
        # plt.show()
        return data.reshape((-1, self.num_channels+1))[:,self.lead_id]

    def get_annotation(self, path, idx):
        row = self.index[self.index.filename_lr == idx]
        labls = [self.morphological_classes[str(l)] for l in row.scp_codes.values[0].keys() if str(l) in self.morphological_classes.keys()] 
        rhytms =[self.rhythmic_classes[str(l)] for l in row.rhythm.values[0] if str(l) in self.rhythmic_classes.keys()] 
        return labls, rhytms

    def extract_wave(self, path, idx):
        """
        Reads .dat file and returns in numpy array. Assumes 2 channels.  The
        returned array is n x 3 where n is the number of samples. The first column
        is the sample number and the second two are the first and second channel
        respectively.
        """
        rdsamp = os.path.join(WFDB, 'rdsamp')
        output = subprocess.check_output([rdsamp, '-r', idx], cwd=path)
        data = np.fromstring(output, dtype=np.int32, sep=' ')
        return data.reshape((-1, self.num_channels+1))
    
    def extract_metadata(self, path, idx):
        infoName=path+os.sep+idx+'.hea'
        fid = open(infoName, 'rt') 
        line = fid.readline() 
        print(line)
        freqint=line.split(" ")
        fs=int(freqint[2])
        self.num_channels = int(freqint[1])
        #print(self.Fs)
        #interval=float(1/self.Fs)

        gains=[]
        bases=[]
        signal_ids=[]

        gains.append(1)
        bases.append(0)
        signal_ids.append("sample")
        nsig=1

        for i in np.arange(0,self.num_channels):
            fields = fid.readline().split(' ')
            gain = fields[2]
            base = fields[4]
            gain = gain.split('/')[0] # in case unit is given
            if len(fields) ==9:
                signal = fields[8]
                signal_ids.append(signal)
                if signal == "II\n" or signal.startswith("MLI"): #look into
                    nsig=i+1
            #[s,s,gain,s,base,s,s,s,signal]
            gains.append(int(gain) if int(gain)!=0 else 200)
            bases.append(int(base))
        fid.close()
        print(signal_ids)
        print(nsig)
        return gains, bases, nsig, fs # nsig is chosen lead

    # def get_class_distributions(self):
    #     Y=self.index
    #     # else:
    #     #Y = pd.read_csv(self.path+os.sep+'ptbxl_database.csv', index_col='ecg_id')
    #     #Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    #     print(Y)
    #     #print(Y.scp_codes)
    #     # Load raw signal data
    #     #X = self.load_raw_data(Y, sampling_rate)

    #     mydict_labels = {}
    #     mydict_rhythms = {}
    #     labels=[]
    #     rhythms=[]

    #     count=0
    #     for ind, row in Y.iterrows():
    #         #if self.classes == 'aami':  
    #         labls = [l for l in row.scp_codes.keys() if l in self.morphological_classes.keys()] 
    #         rhytms = [l for l in row.rhythm if l in self.rhythmic_classes.keys()] 
    #         #print(row.scp_codes.keys())
    #         print(row.rhythm)
    #         # elif self.classes == 'form':
    #         #     labls = row.form
    #         # elif self.classes == 'rhythm':
    #         #     labls = row.rhythm
    #         #print(labls)
    #         labels+=list(labls)
    #         rhythms+=list(rhytms)
    #     #print(labels)
    #     mydict_labels = Counter(labels)
    #     mydict_rhythm = Counter(rhythms)
    #     results_df_rhy = pd.DataFrame.from_dict({"all":mydict_rhythm}, orient='index')
    #     results_df_lab = pd.DataFrame.from_dict({"all":mydict_labels}, orient='index')
    #     #results_df.to_csv(results_path+os.sep+"ptbxl_distribution_"+self.classes+".csv")
    #     return results_df_lab.loc["all",:], results_df_rhy.loc["all",:]

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
            Y = pd.read_csv(self.path+os.sep+'ptbxl_database.csv', index_col='ecg_id')
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