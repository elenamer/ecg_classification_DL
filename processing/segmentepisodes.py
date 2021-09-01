'''

Label aggregation function taken from:
https://github.com/helme/ecg_ptbxl_benchmarking/blob/bed65591f0e530aa6a9cb4a4681feb49c397bf02/code/models/timeseries_utils.py#L534

'''

import numpy as np
from numpy.core.numerictypes import issubdtype
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

def normalize(data):
    data = np.nan_to_num(data)  # removing NaNs and Infs
    std = np.std(data)
    data = data - np.mean(data)
    data = data / std
    if np.std(data)==0:
        print("this again still")
    return data


class SegmentEpisodes():
    def __init__(self, input_size, fs):
        self.input_size = input_size
        self.name = "segmentepisodes"
        self.fs = fs

    def aggregate_labels(self, preds, idmap=None):
        '''
        needs to ba called right after process, meant to be used only in predict function
        '''
        aggregate_fn = np.mean
        print(idmap)
        if idmap is not None:
            print("aggregating predictions...")
            preds_aggregated = []
            targs_aggregated = []
            for i in np.unique(idmap):
                preds_local = preds[np.where(idmap==i)[0]]
                preds_aggregated.append(aggregate_fn(preds_local,axis=0))
            return np.array(preds_aggregated)
        else:
            return np.array(preds)

    def segment_episodes(self, choice, signal, labels_orig, start_minute, end_minute):
        # here start minute is basically start sample

        # ? I don't know for now
        # N_SAMPLES_BEFORE_R_dynamic=int(fs/4.5)
        print(labels_orig.index)

        start_sample = start_minute #int(start_minute * fs * 60)
        start_ind = np.argmax(labels_orig.index >= start_sample)
        if end_minute == -1:
            end_ind = len(signal)
        else: 
            end_sample = end_minute #int(end_minute * fs * 60)
            end_ind = np.argmax(labels_orig.index >= end_sample)

        # print("Start index:")
        # print(start_ind)
        # print("End index:")
        # print(end_ind)
        #print(labels)
        data=[]
        all_labls = []
        labels = pd.DataFrame(labels_orig)
        indices = np.where(np.array(labels.orig_label) == '+')[0]
        print(labels.index[indices])

        for ind, rpeak_ind in enumerate(labels.index[indices]):

            #print(ind, rpeak_ind)
            rPeak_start = rpeak_ind
            #print(labels.loc[ind])
            label=labels.loc[rPeak_start]
            if label.ep_label == '':
                continue
            #print(label)

            if len(indices) > ind+1:
                # print("END DEBUG:")
                # print(ind+1)
                # print(labels.index[indices[ind+1]])
                rPeak_end = labels.index[indices[ind+1]]
            else:
                rPeak_end = len(signal) - 1

            if rPeak_end - rPeak_start > self.input_size * self.fs:
                rPeak_end = rPeak_start + ( self.input_size * self.fs )

            #print(label)
            #print(rPeak)

            sig = signal[rPeak_start:rPeak_end]
            # print("STart: "+str(rPeak_start))
            # print("End: "+str(rPeak_end))

            if np.std(sig)==0:
                print("this happened")
                continue
            #data.append(sig)
            data.append(normalize(sig))
            print(label.ep_label)
            all_labls.append(int(label.ep_label))
            #print(all_labls)

        # print(skipped)
        # print(len(data))
        # print(len(all_labls))
        # print(len(data[0]))
        # print(all_labls)
        # print("All labels")

        return data, all_labls

    def process(self, X, labels=None):
        # input size is the length of a beat in samples
        # labels is encoded index basically
        # ORRR index + either beats_mlb or rhythms_mlb


        # Basically: get all recordings as X (1 row = 1 30 minute signal)
        # Additional argument: lables but in another format
        # Idea: something like R peak locations as additional argument?
        # Return segmented beats, beat-by-beat labels
        # additional arguments needed: such as frequency?(maybe taken from above and implicitly included in input_size), beat length, type of segmentation?
        full_data = []
        full_labels = []
        print(labels)
        choice = "static"
        self.idmap = []
        self.groupmap = []
        for ind, sig in enumerate(X):
            if len(sig) == self.input_size:
                print("no need, already segmented")
                full_data = X
                full_labels = labels
                break
            beats, labls = self.segment_episodes(choice, sig, labels[ind], 0, -1)
            full_data.extend(beats)
            full_labels.extend(labls)
            self.groupmap.extend([ind]*len(beats))
            #print(full_labels)

        lengths = []
        for episode in full_data:
            lengths.append(len(episode)/ self.fs)

        ax = sns.histplot(lengths)
        ax.set_title("Episode duration distribution for "+self.name+" Dataset")
        plt.show()

        self.idmap = np.arange(len(full_data))
        print("after processing")
        print(len(full_data))
        print(len(full_labels))
        print(full_labels[0]) 
        plt.plot(full_data[0])
        plt.show()
        return full_data, full_labels # or maybe groupmap?