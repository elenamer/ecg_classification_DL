
import os
import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter
import subprocess
import wfdb
import numpy as np
import pandas as pd
import pickle
from skimage import transform
import matplotlib.pyplot as plt
from scipy.signal import resample
import tensorflow as tf
from sklearn import preprocessing

random_seed=100

choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

aami_annots_list=['N','L','R','e','j','S','A','a','J','V','E','F','/','f','Q']

WFDB = "/usr/local/bin"#/home/elena/wfdb/bin"


N_SAMPLES_BEFORE_R_static=100
N_SAMPLES_AFTER_R_static=100

N_SAMPLES_BEFORE_R_dynamic=80



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
    return sig # nsig is chosen lead

def segment_beats(choice, ann, signal, beat_len, start_minute, end_minute, fs):
    start_sample = int(start_minute * fs * 60)
    start_ind = np.argmax(ann[:,1] >= start_sample)
    if end_minute == -1:
        end_ind = len(signal)
    else: 
        end_sample = int(end_minute * fs * 60)
        end_ind = np.argmax(ann[:,1] >= end_sample)

    skipped=0
    next_ind=0

    print("Start index:")
    print(start_ind)
    print("End index:")
    print(end_ind)
    data=[]
    labels=[]
    #plt.plot(signal)
    #plt.show()

    for annotation in ann[start_ind:end_ind]:
        rPeak=annotation[1] 
        label=annotation[2]
        next_ind+=1
        #print(label)
        #print(rPeak)
        if not np.isin(label,BEAT_ANNOTATIONS):
            skipped=skipped + 1
            continue

        if choice=="static":
            if rPeak-N_SAMPLES_BEFORE_R_static <0 or rPeak+N_SAMPLES_AFTER_R_static>len(signal):
                continue
            class_label=BEAT_LABEL_TRANSLATIONS[label]
            sig=signal[rPeak-N_SAMPLES_BEFORE_R_static:rPeak+N_SAMPLES_AFTER_R_static]

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
    return data, labels
    #print(len(item)-len(seg_values))

class MITBIHARDataset():

    def __init__(self, name): ## classes, segmentation, selected channel
        self.name = name
        self.path = "./data/"+name+"/"
        #self.num_channels = n_channels
        self.patientids = self.get_patientids()

        self.classes = ["N", "S", "V", "F", "Q"]


        # for patient-specific
        self.common_patients = [101,106,108,109,112,114,115,116,118,119,122,124,100,103,105,111,113,117,121,123]
        self.specific_patients = [201,203,205,207,208,209,215,220,223,230,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

        # for inter-patient
        self.ds1_patients = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
        self.ds2_patients = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

        self.stringify_patientids()

        print(self.patientids)
        #self.X =  # data
        #self.y = 

    def get_patientids(self):
        with open(self.path+"RECORDS"+choice.upper()) as f:
            return f.read().splitlines()
    
    def get_patientids_ds1(self):
        with open(path_to_db+"RECORDS_TEST") as f:
            DS2 = f.read().splitlines()
        with open(path_to_db+"RECORDS_TRAIN") as f:
            DS1 = f.read().splitlines()

    def stringify_patientids(self):
        self.common_patients = [str(id) for id in self.common_patients]
        self.specific_patients = [str(id) for id in self.specific_patients]

        # for inter-patient
        self.ds1_patients = [str(id) for id in self.ds1_patients]
        self.ds2_patients = [str(id) for id in self.ds2_patients]


    '''
    TO DO ELENA: 
        use wfdb python to extrct annotation and wave
    '''

    def extract_metadata(self, idx):
        infoName=self.path+os.sep+idx+'.hea'
        fid = open(infoName, 'rt') 
        line = fid.readline() 
    
        freqint=line.split(" ")
        self.Fs=int(freqint[2])
        self.num_channels = int(freqint[1])
        print(self.Fs)
        interval=float(1/self.Fs)

        gains=[]
        bases=[]
        signal_ids=[]

        gains.append(1)
        bases.append(0)
        signal_ids.append("sample")
        nsig=1

        for i in np.arange(0,self.num_channels):
            [s,s,gain,s,base,s,s,s,signal]=fid.readline().split(' ')
            gains.append(int(gain) if int(gain)!=0 else 1)
            bases.append(int(base))
            signal_ids.append(signal)
            if signal == "II\n" or signal.startswith("MLI"): #look into
                nsig=i+1
        fid.close()
        print(signal_ids)
        print(nsig)
        return gains, bases, nsig # nsig is chosen lead

    def extract_annotation(self, idx):
        """
        The annotation file column names are:
            Time, Sample #, Type, Sub, Chan, Num, Aux
        The Aux is optional, it could be left empty. Type is the beat type and Aux
        is the transition label.
        """
        
        rdann = os.path.join(WFDB, 'rdann')
        output = subprocess.check_output([rdann, '-r', idx, '-a', 'atr'], cwd=self.path)
        labels = (line.split() for line in output.strip().decode().split("\n"))
        new_labels = []
        for l in labels:
            sec_ind = 2 if l[1][-1]==']' else 1
            new_labels.append((l[0], int(l[sec_ind]), l[sec_ind+1], l[sec_ind+2], l[sec_ind+3], l[sec_ind+4], l[sec_ind+5] if len(l) == sec_ind+6 else None))
        #print(len(labels))
        return new_labels

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
        print(self.patientids)
        mydict_labels = {}
        mydict_rhythms = {}
        labels=[]
        rhythms=[]
        for id in self.patientids:
            ann=self.extract_annotation(id)
            ann=np.array(ann)
            print(ann[0])
            annot=ann[:,2]
            if id=='231':
                ## 

                ## How do we handle MISSB annotations?

                ##
                print(ann)
                record = wfdb.rdrecord(self.path+id)
                ann1 = wfdb.rdann(self.path+id, 'atr')

                wfdb.plot_wfdb(record=record, annotation=ann1, title='MIT-BIH Record '+id)
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

    def segment(self, idx, choice, start_minute, end_minute):
        
        gains, bases, nsig = self.extract_metadata(idx)
        print(nsig)
        ecgsig = self.extract_wave(idx)
        sig = remove_base_gain(ecgsig, gains, bases)

        signal = sig[:,nsig]

        annotation = self.extract_annotation(idx)

        ann=np.array(annotation)

        #signal=(signal - np.mean(signal)) / np.std(signal)
        #print(len(annot))
        beat_len = 200
        beats, labels = segment_beats(choice, ann, signal, beat_len, start_minute, end_minute, self.Fs)
        return beats, labels

    def generate_dataset(self,records, choice, balance):
        full_data = []
        full_labels = []
        for patient in self.common_patients:
            beats, labels = self.segment(patient, choice, 0, -1) 
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
        print(len(labels))
        le = preprocessing.LabelEncoder()
        le.fit(self.classes)
        print(list(le.classes_))
        labels = le.transform(labels)
        full_data, full_labels = self.shuffle(data, labels)

        full_labels = tf.keras.utils.to_categorical(full_labels, num_classes=len(self.classes))
        return full_data, full_labels


    def generate_train_set(self, eval_p, choice, balance):
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
            common_data, common_labels = self.generate_dataset(self.common_patients, choice, balance)

            for patient in self.specific_patients:
                full_data = common_data
                full_labels = common_labels
                beats, labels = self.segment(patient, choice, 0, 2.5)
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
            data, labels = self.generate_dataset(self.ds1_patients, choice, balance)
            full_data, full_labels = self.process_dataset(data, labels)
            # mitdb/interpatient/train1_static.pkl
            TRAIN_SET_PATH = path+os.sep+"train1_"+choice+".pkl"
            self.save_dataset(full_data, full_labels, TRAIN_SET_PATH)

        if eval_p == "intra":
            data, labels = self.generate_dataset(self.all_patients, choice, balance)
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
                beats, labels = self.segment(patient, choice, 2.5, 5)
                full_data, full_labels = self.process_dataset(beats, labels)
                # mitdb/specificpatient/val201_static.pkl
                TRAIN_SET_PATH = path+os.sep+"val"+patient+"_"+choice+".pkl"
                self.save_dataset(full_data, full_labels, TRAIN_SET_PATH) 

        if eval_p == 'inter':
            # do nothing for now
            # in this case % train-test-split
            print("not implemented")

    def generate_test_set(self, eval_p, choice, balance=False):
        path = self.path+os.sep+eval_p+"patient"
        if not os.path.exists(path):
            os.makedirs(path)
        if eval_p == "specific":
            for patient in self.specific_patients:
                beats, labels = self.segment(patient, choice, 5, -1)
                full_data, full_labels = self.process_dataset(beats, labels)
                # mitdb/specificpatient/test201_static.pkl
                TRAIN_SET_PATH = path+os.sep+"test"+patient+"_"+choice+".pkl"
                self.save_dataset(full_data, full_labels, TRAIN_SET_PATH) 

        if eval_p == "inter":
            data, labels = self.generate_dataset(self.ds2_patients, choice, balance)
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

