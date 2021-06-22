import os
import pickle
import numpy as np
import pandas as pd
import csv
from collections import Counter
import subprocess
import ast
import fnmatch
import glob
import json
import random
import tqdm

choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

sampling_rate=200

STEP = 256
RELABEL = {"NSR": "SINUS", "SUDDEN_BRADY": "AVB",
           "AVB_TYPE2": "AVB", "AFIB": "AF", "AFL": "AF"}

class iRhythmTestDataset():

    def __init__(self): ## classes, segmentation, selected channel
        self.name = 'irhythm-test'#name
        self.path = "./data/irhythm-test"
        self.patientids = self.get_recordids()
        print(self.patientids)
        #patientids = [os.path.split(id)[-1] for id in patientids]		


    def get_recordids(self, blacklist=set()):
        records = []
        for root, dirnames, filenames in os.walk(self.path):
            for filename in fnmatch.filter(filenames, '*.ecg'):
                if patient_id(filename) not in blacklist:
                    records.append(filename)
        return [id.split('.')[0] for id in records]

    def load_raw_data(self, df, sampling_rate):
        ## implement reading of .ecg files
        return data
    
    def examine_database(self):
        #print(self.patientids)
        mydict_labels = {}
        mydict_rhythms = {}
        labels=[]
        rhythms=[]
        test = load_test(self.path, '_grp*.episodes.json')
        make_json("test.json", test)
        mydict_labels={}
        # load and convert annotation data
        for d in test:
            print(d[1])
            unique_rhythm = Counter(d[1])
            mydict_labels[patient_id(d[0])]=unique_rhythm
        
        results_df=pd.DataFrame.from_dict(mydict_labels, orient='index')
        results_df.loc['all',:] = results_df.sum()
        results_df.to_csv(results_path+os.sep+self.name+"_distribution.csv")

def get_all_records(path, blacklist=set()):
    records = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.ecg'):
            if patient_id(filename) not in blacklist:
                records.append(os.path.abspath(
                    os.path.join(root, filename)))
    return records

def patient_id(record):
    return os.path.basename(record).split("_")[0]

def round_to_step(n, step):
    diff = (n - 1) % step 
    if diff < (step / 2):
        return n - diff
    else:
        return n + (step - diff)

def load_episodes(record, epi_ext):

    base = os.path.splitext(record)[0]
    ep_json = base + epi_ext
    #print(ep_json)
    print(len(glob.glob(ep_json)))
    ep_json = glob.glob(ep_json)[0]

    with open(ep_json, 'r') as fid:
        episodes = json.load(fid)['episodes']
    episodes = sorted(episodes, key=lambda x: x['onset'])

    for episode in episodes:
        episode['onset_round'] = round_to_step(episode['onset'], STEP)
        rn = episode['rhythm_name']
        episode['rhythm_name'] = RELABEL.get(rn, rn)

    for e, episode in enumerate(episodes):
        if e == len(episodes) - 1:
            episode['offset_round'] = episode['offset']
        else:
            episode['offset_round'] = episodes[e+1]['onset_round'] - 1
    return episodes

def make_labels(episodes):
    labels = []
    for episode in episodes:
        rhythm_len = episode['offset_round'] - episode['onset_round'] + 1
        rhythm_labels = int(rhythm_len / STEP)
        rhythm = [episode['rhythm_name']] * rhythm_labels
        labels.extend(rhythm)
    trunc_samp = int(episodes[-1]['offset'] / STEP)
    labels = labels[:trunc_samp]
    return labels

def build_blacklist(blacklist_paths):
    print('Building blacklist...')
    blacklist = set()
    for blacklist_path in blacklist_paths:
        print(blacklist_path)
        for record in get_all_records(blacklist_path):
            blacklist.add(patient_id(record))
    return blacklist

def construct_dataset(records, epi_ext='_grp*.episodes.json'):
    data = []
    for record in tqdm.tqdm(records):
        labels = make_labels(load_episodes(record, epi_ext))
        assert len(labels) != 0, "Zero labels?"
        data.append((record, labels))
    return data

def stratify(records, dev_frac):
    pids = list(set(patient_id(record) for record in records))
    random.shuffle(pids)
    cut = int(len(pids) * dev_frac)
    dev_pids = set(pids[:cut])
    train = [r for r in records if patient_id(r) not in dev_pids] 
    dev = [r for r in records if patient_id(r) in dev_pids] 
    return train, dev 

def load_train(data_path, dev_frac, blacklist_paths):
    blacklist = build_blacklist(blacklist_paths)
    records = get_all_records(data_path, blacklist)
    print(records)
    train, dev = stratify(records, dev_frac)
    print("Constructing train...")
    train = construct_dataset(train)
    print("Constructing dev...")
    dev = construct_dataset(dev)
    return train, dev

def load_rev_id(record, epi_ext):

    base = os.path.splitext(record)[0]
    ep_json = base + epi_ext
    ep_json = glob.glob(ep_json)[0]

    with open(ep_json, 'r') as fid:
        return json.load(fid)['reviewer_id']

def load_test(data_path, epi_ext):
    records = get_all_records(data_path)
    print("Constructing test...")
    test = construct_dataset(records, epi_ext)
    # Get the reviewer id
    reviewers = [load_rev_id(r, epi_ext) for r in records]
    test = [(e, l, r)
            for (e, l), r in zip(test, reviewers)]
    return test

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                    'labels' : d[1]}
            if len(d) == 3:
                datum['reviewer'] = d[2]
            json.dump(datum, fid)
            fid.write('\n')
