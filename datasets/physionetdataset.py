#x:(n_samples, beat_length)
#y:(n_samples)

import os
import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter
import subprocess
import wfdb

# Basically: generate per-patient files, with segmented beats and labels
# these files are an input in datagenerator class, which concatenates e.g. train and test patients kako sto treba
# (not exactly, this way they would all be in the same batch and we don't want that)
choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

aami_annots_list=['N','L','R','e','j','S','A','a','J','V','E','F','/','f','Q']

WFDB = "/usr/local/bin"#/home/elena/wfdb/bin"

classes = ['N', 'S', 'V', 'F', 'Q']


class PhysionetDataset():

    def __init__(self, name, n_channels): ## classes, segmentation, selected channel
        self.name = name
        self.path = "./data/"+name+"/"
        self.num_channels = n_channels
        self.patientids = self.get_patientids()
        print(self.patientids)
            #patientids = [os.path.split(id)[-1] for id in patientids]		

    def get_patientids(self):
        with open(self.path+"RECORDS"+choice.upper()) as f:
            return f.read().splitlines()
    '''
    TO DO ELENA: 
        use wfdb python to extrct annotation and wave

    '''

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
