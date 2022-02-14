
import numpy as np
from numpy.core.numerictypes import issubdtype
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import os

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
        #print(labels.index[indices])

        for ind, rpeak_ind in enumerate(labels.index[indices]):

            #print(ind, rpeak_ind)
            rPeak_start = rpeak_ind
            #print(labels.loc[ind])
            label=labels.loc[rPeak_start]
            #print(label)
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

            print(rPeak_end)
            print(rPeak_start)
            print(self.input_size)
            if rPeak_end - rPeak_start > self.input_size:
               rPeak_end = rPeak_start + ( self.input_size)
            print(rPeak_end)

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
            

            # uncomment for episodes visualization

            # if int(label.ep_label) != 0:
            #     fig, ax = plt.subplots()               
            #     ax.plot(normalize(sig))
            #     ax.annotate(int(label.ep_label), xy=(0, sig[0]), xycoords='data')
            #     plt.show()

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

        ## uncomment for duration distributions per class

        # distrs_splits = {}
        # for ind, episode in enumerate(full_data):

        #     key = int(full_labels[ind])
            
        #     if key in distrs_splits:
        #         distrs_splits[key].append(len(episode)/ self.fs)
        #     else:
        #         distrs_splits[key] = []
        # for ind, key in enumerate(distrs_splits.keys()):
        #     ax = sns.kdeplot(distrs_splits[key], shade=True, label = key)
        #     if ind>len(list(distrs_splits.keys()))-2:
        #         break
        #     for key2 in list(distrs_splits.keys())[ind+1:]:
        #         if len(distrs_splits[key2]) == 0:
        #             continue
        #         # text_file.write("pair: "+str(key)+" "+str(key2)+"\n")
        #         # text_file.write(str(scipy.stats.ks_2samp(distrs_splits[key],distrs_splits[key2]))+"\n")

        # ax.legend()
        # ax.set_title("Episode duration distribution for "+self.name+" Dataset")
        # plt.show()

        self.idmap = np.arange(len(full_data))
        print("after processing")
        print(len(full_data))
        print(len(full_labels))
        print(full_labels[0]) 
        plt.plot(full_data[0])
        plt.show()
        return full_data, full_labels # or maybe groupmap?