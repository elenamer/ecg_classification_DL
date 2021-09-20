#x:(n_samples, beat_length)
#y:(n_samples)

from processing.segmentbeats import SegmentBeats
from datasets.dataset import Dataset
import os
import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter
import subprocess
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import itertools

# Basically: generate per-patient files, with segmented beats and labels
# these files are an input in datagenerator class, which concatenates e.g. train and test patients kako sto treba
# (not exactly, this way they would all be in the same batch and we don't want that)

random_seed=100


choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

aami_annots_list=['N','L','R','e','j','S','A','a','J','V','E','F','/','f','Q']

WFDB = "/usr/local/bin"#/home/elena/wfdb/bin"



BEAT_ANNOTATIONS=["N","L","R","e","j","A","a","J","S","V","E","F","Q","/","f"]
BEAT_LABEL_TRANSLATIONS={
"N":"N","L":"N","R":"N","e":"N","j":"N", 
"A":"S","a":"S","J":"S","S":"S", 
    "V":"V","E":"V",  
    "F":"F", 
"Q":"Q","/":"Q","f":"Q"
}

def normalize(data):
    data = np.nan_to_num(data)  # removing NaNs and Infs
    std = np.std(data)
    data = data - np.mean(data)
    data = data / std
    if np.std(data)==0:
        print("this again still")
    return data

def remove_base_gain(ecgsig, gains, bases):
    sig=ecgsig.astype(np.float32)
    sig[sig == - 32768]=np.nan
    gains=np.array(gains)
    bases=np.array(bases)
    for i in np.arange(0,np.size(ecgsig,1)):
        sig[:,i]=(sig[:,i] - bases[i]) / gains[i]
    return sig 
    
def segment_beats(choice, ann, signal, beat_len, start_minute, end_minute, fs):

    N_SAMPLES_BEFORE_R_static=int(fs/3.6)
    N_SAMPLES_AFTER_R_static=int(fs/3.6)

    N_SAMPLES_BEFORE_R_dynamic=int(fs/4.5)

    print(N_SAMPLES_BEFORE_R_static)
    print(N_SAMPLES_BEFORE_R_dynamic)

    start_sample = int(start_minute * fs * 60)
    start_ind = np.argmax(ann[:,1] >= start_sample)
    if end_minute == -1:
        end_ind = len(signal)
    else: 
        end_sample = int(end_minute * fs * 60)
        end_ind = np.argmax(ann[:,1] >= end_sample)

    skipped=0
    next_ind=start_ind
    print("Start index:")
    print(start_ind)
    print("End index:")
    print(end_ind)
    data=[]
    labels=[]
    #plt.plot(signal)
    #plt.show()
    all_labls = []

    for annotation in ann[start_ind:end_ind]:
        rPeak=annotation[1] 
        label=annotation[2]
        next_ind+=1
        all_labls.append(label)
        #print(label)
        #print(rPeak)
        if not np.isin(label,BEAT_ANNOTATIONS):
            skipped=skipped + 1
            continue

        if choice=="static":
            if rPeak-N_SAMPLES_BEFORE_R_static <0 or rPeak+N_SAMPLES_AFTER_R_static>len(signal):
                continue
            class_label=BEAT_LABEL_TRANSLATIONS[label]
            sig=resample(signal[rPeak-N_SAMPLES_BEFORE_R_static:rPeak+N_SAMPLES_AFTER_R_static], beat_len)

        else:#if choice=="dynamic": 
            if rPeak-N_SAMPLES_BEFORE_R_dynamic <0:
                continue        
            if len(ann[:,2]) == next_ind:
                sig=resample(signal[rPeak-N_SAMPLES_BEFORE_R_dynamic:],beat_len)
            else:
                rPeak_next=ann[next_ind,1]
                print(signal)
                print(rPeak-N_SAMPLES_BEFORE_R_dynamic)
                print(rPeak_next-N_SAMPLES_BEFORE_R_dynamic)
                sig=resample(signal[rPeak-N_SAMPLES_BEFORE_R_dynamic:rPeak_next-N_SAMPLES_BEFORE_R_dynamic],beat_len)
            class_label=BEAT_LABEL_TRANSLATIONS[label]
        if np.std(sig)==0:
            print("this happened")
            continue
        #plt.plot(sig)
        #plt.show()
        data.append(normalize(sig))
        #seg_values.append(sig)
        labels.append(class_label)
        #print(seg_values)

    #print(len(ann[:,1])-len(seg_values))
    print(skipped)
    print(len(data))
    print(len(labels))
    print(len(data[0]))
    print(labels)
    print("All labels")
    print(Counter(all_labls))
    return data, labels
    #print(len(item)-len(seg_values))


class PhysionetDataset(Dataset):

    def __init__(self, name, task, eval): ## classes, segmentation, selected channel
        
        super(PhysionetDataset, self).__init__(task, eval)

        self.name = name

        self.path = "./data/"+name+"/"
        self.common_path = "./data/"+name+"/"
        self.patientids = self.get_patientids()
        print(self.patientids)
        print(self.common_path)

        self.patient_groups = self.get_patientgroups()

        #self.k_fold_crossval()
        #patientids = [os.path.split(id)[-1] for id in patientids]		

    def get_patientids(self):
        with open(self.path+"RECORDS"+choice.upper()) as f:
            return f.read().splitlines()

    def get_patientgroups(self):
        patientgroups_dict={}
        with open(self.path+"files-patients-diagnoses.txt") as f:
            lines = f.read().splitlines()
        for (ind, line) in enumerate(lines):
            if line.startswith("patient"):
                number = line.split(" ")[1]
                print(number)
                patientids = lines[ind+1].split(" ")
                print(patientids)
                for patient in patientids:
                    patientgroups_dict[patient] = number
        return patientgroups_dict

    def get_patientids_ds1(self):
        with open(path_to_db+"RECORDS_TEST") as f:
            DS2 = f.read().splitlines()
        with open(path_to_db+"RECORDS_TRAIN") as f:
            DS1 = f.read().splitlines()

    def generate_counts(self):
        transf = SegmentBeats(input_size = int(0.72 * self.freq))
        X, y = self.get_data()
        X, y, idmap = transf.process(X, labels=y)
        groupmap = transf.groupmap
        temp = pd.DataFrame(y, index=groupmap)
        temp = temp.groupby(level=0).sum()
        temp[0] = temp.sum(axis=1)
        print("tamp values print")
        print(temp.values.tolist())
        return temp.values

    def generate_counts1(self):
        transf = SegmentBeats(input_size = int(0.72 * self.freq))
        X, y = self.get_data()
        X, y, idmap = transf.process(X, labels=y)
        groupmap = transf.groupmap
        temp = pd.DataFrame(y, index=groupmap)
        temp = temp.groupby(level=0).sum()
        temp[0] = temp.sum(axis=1)
        return X, y, groupmap


    '''

    HAve something like this:

    include all functions from ptb xl here:

    have some kind of an index file with labels and e.g. 10s signals (identified by e.g. patient id and start and end sample)


    '''

    def encode_labels(self, ann):
        '''
        
        it's a problem that encode_labels is different than in other datasets
        
        '''
        mlb_rhy = MultiLabelBinarizer(classes=range(len(self.class_names.values)))

        #encoded_index.set_index("FileName", inplace=True)
        print(len(set(ann.sample)))
        print(len(ann.sample))
        indices = sorted(dict(zip(reversed(ann.sample), range(len(ann.sample)-1, -1, -1))).values())
        ann.sample = list(np.array(ann.sample)[indices])
        ann.symbol = list(np.array(ann.symbol)[indices])
        ann.aux_note = list(np.array(ann.aux_note)[indices])

        encoded_index = pd.DataFrame(np.column_stack((ann.symbol, ann.aux_note)), columns=["orig_label","ep_label"], index = ann.sample)
        encoded_index["labels_mlb"] = ""
        encoded_index["episodes_mlb"] = ""

        for ind, sample_ind in enumerate(ann.sample):
            #print(row)        
            rhythms =[self.classes[str(l)] for l in [encoded_index.at[sample_ind, "orig_label"]] if str(l) in self.classes.keys()] 
            #print(str(encoded_index.at[sample_ind, "ep_label"]))
            if str(encoded_index.at[sample_ind, "ep_label"]) in self.classes.keys():
                episode = self.classes[str(encoded_index.at[sample_ind, "ep_label"])]
                if episode != 0:
                    #print("EPPPPPP")
                    #print(self.classes)
                    print(episode)
            else:
                episode = ''

            encoded_index.at[sample_ind, "labels_mlb"] = tuple(rhythms)
            encoded_index.at[sample_ind, "ep_label"] = episode

        encoded_index["labels_mlb"] = mlb_rhy.fit_transform(encoded_index["labels_mlb"]).tolist()
        #print(encoded_index["labels_mlb"])
        #encoded_index = encoded_index[["labels_mlb"]]
        #print(encoded_index)

        #encoded_index = encoded_index[(encoded_index.labels_mlb.apply(lambda x:sum(x)) > 0).values]
        #print(encoded_index.ep_label.values)
        return encoded_index


    def get_signal(self, path, idx):
        data, metadata = wfdb.rdsamp(path+idx)
        #print(metadata['sig_name'])
        lead_names = self.lead.split("-")
        if len(lead_names) == 1:
            if lead_names[0] in metadata['sig_name']:
                lead_ind = metadata['sig_name'].index(lead_names[0])
            elif ("ML"+lead_names[0]) in metadata['sig_name']:
                lead_ind = metadata['sig_name'].index("ML"+lead_names[0])
            else:
                lead_ind = 0
            sig = data[:,lead_ind]
        else:
            #  should do the same here 
            sig = data[:,metadata['sig_name'].index(lead_names[0])] - data[:,metadata['sig_name'].index(lead_names[1])]
        return sig


    def get_annotation(self, path, idx):
        ann = wfdb.rdann(path+idx, "atr")

        '''
        
        keep it like this for now, but it's a problem that get_annotation returns different things than in other datasets
        
        '''
        print(ann.symbol)
        print([l[:-1] for l in ann.aux_note])
        encoded_index = self.encode_labels(ann)
        return encoded_index


    def get_data(self):
        # only to be used for optimizer for now, should be changed/improved
        max_size=100 # FOr now, should remove this
        # Load PTB-XL data
        X = [self.get_signal(self.path,id) for id in self.patientids[:max_size]]

                       
        X=np.array(X)

        y = [self.get_annotation(self.path,id) for id in self.patientids[:max_size]]
        
        '''
            old: self.encoded_labels.iloc[:max_size,:]

            this here needs to be custom to physionet 

        '''    
        
        print("before")
        # Preprocess label data

        #y = [df["labels_mlb"] for df in y]

        print(X.shape)

        # Preprocess signal data
        #self.X_train, self.X_val, self.X_test = preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        # self.n_classes = self.y_train.shape[1]
        # partition = {"train": self.y_test.filename_lr.values.tolist() ,"validation":self.y_test.filename_lr.values.tolist(), "test":self.y_test.filename_lr.values.tolist()}
        print(len(y))# print(y_test.shape)
        # print(y_train.shape)
        # print(y_val.shape)
        return X, y


    def get_split_interpatient(self, split=9):
        ## this is for interpatient
        max_size=100 # FOr now, should remove this

        X_train = [self.get_signal(self.path,id) for id in self.ds1_patients_train[:max_size]]
        X_test = [self.get_signal(self.path,id) for id in self.ds2_patients[:max_size]]
        X_val = [self.get_signal(self.path,id) for id in self.ds1_patients_val[:max_size]]
                       
        X_train=np.array(X_train)
        X_test=np.array(X_test)
        X_val=np.array(X_val)


        y_train = [self.get_annotation(self.path,id) for id in self.ds1_patients_train[:max_size]]
        y_test = [self.get_annotation(self.path,id) for id in self.ds2_patients[:max_size]]
        y_val = [self.get_annotation(self.path,id) for id in self.ds1_patients_val[:max_size]]
    
        
        print("before")
        # Preprocess label data
        print(y_train[0])
        #y_train = [df["episodes_mlb"] for df in y_train]
        #y_test = [df["episodes_mlb"] for df in y_test]
        #y_val = [df["episodes_mlb"] for df in y_val]

        print(X_test.shape)
        print(X_val.shape)

        # Preprocess signal data
        #self.X_train, self.X_val, self.X_test = preprocess_signals(self.X_train, self.X_val, self.X_test, self.outputfolder+self.experiment_name+'/data/')
        # self.n_classes = self.y_train.shape[1]
        # partition = {"train": self.y_test.filename_lr.values.tolist() ,"validation":self.y_test.filename_lr.values.tolist(), "test":self.y_test.filename_lr.values.tolist()}
        print(len(y_test))# print(y_test.shape)
        # print(y_train.shape)
        # print(y_val.shape)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_overall_patientgroups(self, groups):
        overall_groups = np.copy(groups)
        for ind, patient in enumerate(self.patientids):
            overall_groups[overall_groups==ind] = self.patient_groups[patient]
        overall_groups = np.array(overall_groups)
        print(np.unique(overall_groups, return_counts=True)[1])
        print(np.unique(groups, return_counts=True)[1])
        return overall_groups

    def get_crossval_splits(self, X, Y, recording_groups, split=9):
        ## this is for interpatient
        
        indices = np.sum(np.array(Y), axis=1) > 0 
        print(X)
        data = np.array(X)[indices]
        labels = np.array(Y[indices])
        recording_groups = np.array(np.array(recording_groups)[indices])
        
        overall_groups = self.get_overall_patientgroups(recording_groups)

        train, test= next(itertools.islice(self.strat_group_k_fold.split(data,labels.argmax(1),overall_groups), split, None))
        X_test, y_test = data[test], labels[test]
        if split != 0:
            val_split = split - 1
        else:
            val_split = self.strat_group_k_fold.n_splits - 1
        # for now always have a different validation set
        train, val= next(itertools.islice(self.strat_group_k_fold.split(data,labels.argmax(1),overall_groups), val_split, None))
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
        print(len(y_test))# print(y_test.shape)
        # print(y_train.shape)
        # print(y_val.shape)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_crossval_splits_intrapatient(self, X, Y, split=9):
        ## this is for interpatient
        
        indices = np.sum(np.array(Y), axis=1) > 0 
        print(X)
        data = np.array(X)[indices]
        labels = np.array(Y[indices])


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
        print(len(y_test))# print(y_test.shape)
        # print(y_train.shape)
        # print(y_val.shape)
        return X_train, y_train, X_val, y_val, X_test, y_test


    '''
    TO DO ELENA: 
        use wfdb python to extrct annotation and wave
    '''

    def get_class_distributions(self, list_classes):
        print(self.patientids)
        mydict_labels = {}
        mydict_rhythms = {}
        labels=[]
        rhythms=[]
        for id in self.patientids:
            #print(id)
            ann=self.extract_annotation(self.path, id)
            ann=np.array(ann)
            #print(ann[0])
            annot=ann[:,2]
            annot = [self.classes[a] for a in annot if a in self.classes.keys()]
            unique_labels = Counter(annot)

            rhythm_labels = ann[:,6]
            #print(rhythm_labels)
            rhythm_labels = [self.classes[l] for l in rhythm_labels if l in self.classes.keys()]

            unique_rhythm = Counter(rhythm_labels)
            unique_labels.update(unique_rhythm)
            ind_seg=0
            mydict_labels[id]=unique_labels
            #print(mydict_rhythms)
            labels+=list(unique_labels.keys())
        labels=set(labels)
        results_df_lab=pd.DataFrame.from_dict(mydict_labels, orient='index')
        results_df_lab.loc['all',:] = results_df_lab.sum()
        #results_df_lab.to_csv(results_path+os.sep+self.name+"_morphological_distribution.csv")

        return results_df_lab.loc['all',:]

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
        #print(signal_ids)
        #print(nsig)
        return gains, bases, nsig, fs # nsig is chosen lead

    def extract_annotation(self, path, idx):
        """
        The annotation file column names are:
            Time, Sample #, Type, Sub, Chan, Num, Aux
        The Aux is optional, it could be left empty. Type is the beat type and Aux
        is the transition label.
        """
        
        rdann = os.path.join(WFDB, 'rdann')
        output = subprocess.check_output([rdann, '-r', idx, '-a', 'atr'], cwd=path)
        labels = (line.split() for line in output.strip().decode().split("\n"))
        new_labels = []
        for l in labels:
            sec_ind = 2 if l[1][-1]==']' else 1
            new_labels.append((l[0], int(l[sec_ind]), l[sec_ind+1], l[sec_ind+2], l[sec_ind+3], l[sec_ind+4], l[sec_ind+5] if len(l) == sec_ind+6 else None))
        print(len(new_labels))
        #print(len(labels))
        return new_labels

    def extract_wave(self, path, idx):
        """
        Deprecated maybe
        """
        rdsamp = os.path.join(WFDB, 'rdsamp')
        output = subprocess.check_output([rdsamp, '-r', idx], cwd=path)
        data = np.fromstring(output, dtype=np.int32, sep=' ')
        return data.reshape((-1, self.num_channels+1))

    def examine_database(self):
        print(self.patientids)
        mydict_labels = {}
        mydict_rhythms = {}
        labels=[]
        rhythms=[]
        for id in self.patientids:
            print(id)
            ann=self.extract_annotation(self.path, id)
            ann=np.array(ann)
            print(ann[0])
            annot=ann[:,2]
            if id=='231':
                ## 

                ## How do we handle MISSB annotations?

                ##
                print(ann)
                #record = wfdb.rdrecord(self.path+id)
                #ann1 = wfdb.rdann(self.path+id, 'atr')

                #wfdb.plot_wfdb(record=record, annotation=ann1, title='MIT-BIH Record '+id)
            #print(ann[0])
            unique_labels = Counter(annot)

            rhythm_labels = ann[:,6]
            rhythm_labels = [l for l in rhythm_labels if l is not None ]
            unique_rhythm = Counter(rhythm_labels)
            unique_labels.update(unique_rhythm)
            
            # AAMI Classes:
            # # N = N, L, R, e, j
            # # S = A, a, J, S
            # # V = V, E
            # # F = F
            # # Q = /, f, Q

            ind_seg=0
            mydict_labels[id]=unique_labels
            #mydict_rhythms[id]=unique_rhythm
            labels+=list(unique_labels.keys())
            #rhythms+=list(unique_rhythm.keys())
        print(labels)
        labels=set(labels)
        #rhythms=set(rhythms)
        results_df=pd.DataFrame.from_dict(mydict_labels, orient='index')
        #results_df.columns=labels
        #results_df_lab = pd.DataFrame(columns = labels, index=self.patientids +["all"] )
        #results_df_rhy = pd.DataFrame(columns = rhythms, index=self.patientids +["all"] )
        
        for l in aami_annots_list[::-1]:
            if l in results_df.columns:
                col = results_df.pop(l)
                results_df.insert(0, col.name, col)

        results_df.loc['all',:] = results_df.sum()
        results_df.to_csv(results_path+os.sep+self.name+"_distribution.csv")

    def segment(self, path, idx, choice, start_minute, end_minute):
        
        gains, bases, nsig, fs = self.extract_metadata(path, idx)
        print(nsig)
        ecgsig = self.extract_wave(path, idx)
        sig = remove_base_gain(ecgsig, gains, bases)

        signal = sig[:,nsig]

        annotation = self.extract_annotation(path, idx)

        ann=np.array(annotation)

        #signal=(signal - np.mean(signal)) / np.std(signal)
        #print(len(annot))
        beat_len = 200
        beats, labels = segment_beats(choice, ann, signal, beat_len, start_minute, end_minute, fs)
        return beats, labels

    def generate_dataset(self, path, records, choice, balance):
        full_data = []
        full_labels = []
        for patient in records:
            beats, labels = self.segment(path, patient, choice, 0, -1) 
            full_data.extend(beats)
            full_labels.extend(labels)
        if balance:
            full_data, full_labels = self.balance(full_data, full_labels)
        return full_data, full_labels

    def balance(self, data, labels):
        # To Do: add more options for balancing, which classes and how much
        print(Counter(labels))
        labels = np.array(labels)
        data = np.array(data)
        C0 = np.argwhere(labels == 'N').flatten()
        C0 = np.concatenate((C0, np.argwhere(labels==0).flatten()), axis=0)
        C0_subsampled = C0[0::10]
        C0 = np.setdiff1d(C0,C0_subsampled)
        print("N indices")
        print(len(C0))
        print(len(C0_subsampled))
        labels = np.delete(labels,C0,axis=0)
        data = np.delete(data,C0,axis=0) 
        print(Counter(labels))
        return data.tolist(), labels.tolist()

    def shuffle(self, data, labels):
        #np.random.seed(random_seed)
        permute = np.random.permutation(len(labels))
        data= data[permute]
        labels = labels[permute]
        return data, labels

    def process_dataset(self, data, labels):
        data = np.asarray(data)
        print(data.shape)

        #data=data[:,:,0]
        #full_labels = np.array(labels)
        cl_dict = {}
        for i, cl in  enumerate(self.classes):
            cl_dict[cl] = i
        labels = [cl_dict[l] for l in labels]
        print(Counter(labels))
        labels = np.array(labels)
        full_data, full_labels = self.shuffle(data, labels)
        #full_labels = tf.keras.utils.to_categorical(full_labels, num_classes=len(self.classes))
        return full_data, full_labels


    def generate_train_set(self, eval_p, choice, balance, full=False):
        # for patient in patients(train):
        #     self.segment(idx) -> full
        # if intra:
        #     for patient in patients(train and test):
        #         self.segment(idx) -> full
        #         np.random()
        path = self.path+os.sep+eval_p+"patient"
        if not os.path.exists(path):
            os.makedirs(path)
        if eval_p == "specific":
            common_data, common_labels = self.generate_dataset(self.common_path, self.common_patients, choice, balance)

            for patient in self.specific_patients:
                full_data = common_data
                full_labels = common_labels
                beats, labels = self.segment(self.path, patient, choice, 0, 2.5)
                if balance:
                    beats, labels = self.balance(beats, labels)
                print(len(full_data))
                print(len(full_labels))
                print(len(beats))
                print(len(labels))
                full_data.extend(beats)
                full_labels.extend(labels)
                print(len(full_data))
                print(len(full_labels))
                # mitdb/specificpatient/train201_static.pkl
                TRAIN_SET_PATH = path+os.sep+"train"+patient+"_"+choice+".pkl"
                full_data, full_labels = self.process_dataset(full_data, full_labels)
                self.save_dataset(full_data, full_labels, TRAIN_SET_PATH)

        if eval_p == "inter":
            if full:
                patients = self.ds1_patients_train+self.ds1_patients_val
            else:
                patients = self.ds1_patients_train
            data, labels = self.generate_dataset(patients, choice, balance)
            full_data, full_labels = self.process_dataset(data, labels)
            # mitdb/interpatient/train1_static.pkl
            TRAIN_SET_PATH = path+os.sep+"train1_"+choice+".pkl"
            self.save_dataset(full_data, full_labels, TRAIN_SET_PATH)

        if eval_p == "intra":
            data, labels = self.generate_dataset(self.path, self.patientids, choice, balance)
            full_data, full_labels = self.process_dataset(data, labels)
            # mitdb/interpatient/train_static.pkl
            TRAIN_SET_PATH = path+os.sep+"train_"+choice+".pkl"
            self.save_dataset(full_data, full_labels, TRAIN_SET_PATH)    

    def generate_val_set(self, eval_p, choice, balance=False):
        path = self.path+os.sep+eval_p+"patient"
        if not os.path.exists(path):
            os.makedirs(path)
        if eval_p == 'specific':
            for patient in self.specific_patients:
                beats, labels = self.segment(self.path, patient, choice, 2.5, 5)
                full_data, full_labels = self.process_dataset(beats, labels)
                # mitdb/specificpatient/val201_static.pkl
                TRAIN_SET_PATH = path+os.sep+"val"+patient+"_"+choice+".pkl"
                self.save_dataset(full_data, full_labels, TRAIN_SET_PATH) 

        if eval_p == "inter":
            data, labels = self.generate_dataset(self.path, self.ds1_patients_val, choice, balance)
            full_data, full_labels = self.process_dataset(data, labels)
            # mitdb/interpatient/test1_static.pkl
            TRAIN_SET_PATH = path+os.sep+"val1_"+choice+".pkl"
            self.save_dataset(full_data, full_labels, TRAIN_SET_PATH) 

    def generate_test_set(self, eval_p, choice, balance=False):
        path = self.path+os.sep+eval_p+"patient"
        if not os.path.exists(path):
            os.makedirs(path)
        if eval_p == "specific":
            for patient in self.specific_patients:
                beats, labels = self.segment(self.path, patient, choice, 5, -1)
                full_data, full_labels = self.process_dataset(beats, labels)
                # mitdb/specificpatient/test201_static.pkl
                TRAIN_SET_PATH = path+os.sep+"test"+patient+"_"+choice+".pkl"
                self.save_dataset(full_data, full_labels, TRAIN_SET_PATH) 

        if eval_p == "inter":
            data, labels = self.generate_dataset(self.path, self.ds2_patients, choice, balance)
            full_data, full_labels = self.process_dataset(data, labels)
            # mitdb/interpatient/test1_static.pkl
            TRAIN_SET_PATH = path+os.sep+"test1_"+choice+".pkl"
            self.save_dataset(full_data, full_labels, TRAIN_SET_PATH) 
        if eval_p == "intra":
            # in this case % train-test-split
            print("not implemented")


    def save_dataset(self, data, labels, path):
        with open(path, 'wb') as file:
            pickle.dump({'data': data, 'labels': labels}, file)  

    def load_dataset(self, eval_p, choice, dataset, crossval_id, balance=True):
        # mitdb/interpatient/test1_static.pkl

        path = self.path + os.sep + eval_p + 'patient' + os.sep + dataset + crossval_id + "_" + choice+".pkl"
        print(path)
        if not os.path.isfile(path):
            return False
        with open(path,"rb") as fid:
            dictionary=pickle.load(fid)
        data = dictionary['data']
        labels = dictionary['labels']
        return data, labels

