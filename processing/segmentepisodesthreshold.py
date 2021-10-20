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

def func(row):
    labls = []
    labls.append(row.ep_label)
    if sum(row.labels_mlb) != 0:
        #print(row.labels_mlb)
        l = np.array(row.labels_mlb).argmax()
        #print(lls)
        #print(l)
        labls.append(l)
    #print(labls)
    #print(labls)
    return labls

def get_episode_label(labels):
    all_labels = []
    for ind, row in labels.iterrows():
        l = func(row)
        all_labels.extend(l)
    values, counts = np.unique(np.array(all_labels), return_counts = True)
    to_drop = np.where(values == '')[0]
    #print(values)
    #print(counts)
    values = np.delete(values, to_drop)
    counts = np.delete(counts, to_drop)
    if counts.size > 0:
        print(values)
        # if counts.size > 1:
        #     to_drop = np.where(values == '0')[0]
        #     values = np.delete(values, to_drop)
        #     counts = np.delete(counts, to_drop)
        # chosen_label_index = counts.argmax()
        # chosen_label = values[chosen_label_index]
        # print(chosen_label)
        # # if str(chosen_label) != '0' and str(chosen_label) != '':
        # #     print(values)
        # #     print(counts)
    else:
        chosen_label = ''
        return [chosen_label]
    #if counts[chosen_label_index] < 4:
    #    return ''
    #all_labels.append(labels.ep_label.values)
    #all_labels.append(labels.ep_label.values.argmax(axis=1))
    return values



class SegmentEpisodesThreshold():
    def __init__(self, input_size, fs):
        self.input_size = input_size
        self.name = "segmentepisodesthreshold"
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
        '''
        
        Main idea: have realistic windows. for real. not ''edited'' according to class in any way.
        
        '''
        
        # here start minute is basically start sample

        print(labels_orig.index)

        start_sample = start_minute #int(start_minute * fs * 60)
        start_ind = np.argmax(labels_orig.index >= start_sample)
        if end_minute == -1:
            end_sample = len(signal)
        else:
            end_sample = end_minute #int(end_minute * fs * 60)

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

            current_ep = labels.loc[rpeak_ind].ep_label

            print(current_ep)
            # if current_ep == '':
            #     print("VERYYY IMPORTANT")
            #print(ind, rpeak_ind)
            rPeak_start = rpeak_ind
            
            if len(indices) > ind+1:
                # print("END DEBUG:")
                # print(ind+1)
                # print(labels.index[indices[ind+1]])
                rPeak_end = labels.index[indices[ind+1]]
            else:
                rPeak_end = len(signal) - 1

            for ind, label in labels.loc[rPeak_start:rPeak_end].iterrows():#range(rPeak_start,rPeak_end):
                #label = labels.loc[ind]

                if label.ep_label == '':
                    label.ep_label = current_ep
            #if current_ep == '':
                #print(labels.loc[rPeak_start:rPeak_end])

            #print(labels.loc[rPeak_start:rPeak_end])
            #print(label)

        #input()

        for window_start in range(start_sample, end_sample, self.input_size):
            
            window_end = window_start+self.input_size
            #print(self.input_size / 360)
            start_ind = np.argmax(labels_orig.index >= window_start)


            '''
            problematic end ind
            
            '''
            if np.argmax(labels_orig.index >= window_end) != 0:
                end_ind = np.argmax(labels_orig.index >= window_end)
            else:
                end_ind = len(signal)



            #print("window: "+str(window_start)+"   "+str(window_end))
            #print("index: "+str(start_ind)+"   "+str(end_ind))
            episode_labels=labels.iloc[start_ind:end_ind]
            #print(labels.iloc[start_ind:end_ind])
            label = get_episode_label(episode_labels)

            if label[0] == '':
                continue
            
            #print(label)

            sig = signal[window_start:window_end]

            if np.std(sig)==0:
                print("this happened")
                continue
            #data.append(sig)
            data.append(normalize(sig))
            #print(label)
            all_labls.extend(label)
            #print(all_labls)
            #plt.plot(normalize(sig))

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

        ax = sns.histplot(full_labels, stat='probability')
        ax.set_title("Episode label distribution for "+self.name+" Dataset")
        #plt.show()
        for container in ax.containers:
            ax.bar_label(container)
        plt.savefig("./"+self.name+"_ep_distr_mlb.png", dpi=300)

        self.idmap = np.arange(len(full_data))
        print("after processing")
        print(len(full_data))
        print(len(full_labels))
        print(full_labels[0]) 
        plt.plot(full_data[0])
        plt.show()
        return full_data, full_labels # or maybe groupmap?