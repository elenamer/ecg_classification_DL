
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
        l = np.array(row.labels_mlb).argmax()
        labls.append(l)
    return labls

def get_episode_label(labels, end_index):
    all_labels = []
    #print("labels")
    #print(labels)

    cnts = {}
    next_ind = 1
    for row in labels.itertuples():
        current = row.Index # current sample
        try:
            next = labels.iloc[next_ind].Index # next sample
        except:
            next = end_index
        current_length = next-current
        l = func(row)
        all_labels.extend(l)
        for label_ind in l:
            if label_ind in cnts.keys():
                cnts[label_ind] += current_length
            else:
                cnts[label_ind] = current_length
        next_ind+=1
    temp = dict(sorted(cnts.items(), key=lambda item: item[1]))
    values = list(temp.keys())
    counts = [temp[k] for k in temp.keys()]

    values, counts = np.unique(np.array(all_labels), return_counts = True)
    to_drop = np.where(np.array(values) == '')[0]
    values = np.delete(values, to_drop)
    counts = np.delete(counts, to_drop)
    if counts.size > 0:
        if counts.size > 1:
            to_drop = np.where(values == '0')[0]
            values = np.delete(values, to_drop)
            counts = np.delete(counts, to_drop)
        chosen_label_index = counts.argmax()
        chosen_label = values[chosen_label_index]
    else:
        chosen_label = ''
    #    return [chosen_label]
    # if counts[chosen_label_index] < 4:
    #    return ''
    all_labels.append(labels.ep_label.values)
    #all_labels.append(labels.ep_label.values.argmax(axis=1)) # (if we want argmax)
    # return values # (if we want mlb)
    return chosen_label


class SegmentEpisodesThreshold():
    def __init__(self, input_size, fs):
        self.input_size = input_size
        self.name = "segmentepisodesthreshold"
        self.fs = fs

    def segment_episodes(self, choice, signal, labels_orig, start_minute, end_minute):
        '''
        
        Main idea: have realistic windows. not ''edited'' according to class in any way.
        
        '''
        
        # here start minute is basically start sample

        print(labels_orig.index)

        start_sample = start_minute #int(start_minute * fs * 60)
        start_ind = np.argmax(labels_orig.index >= start_sample)
        if end_minute == -1:
            end_sample = len(signal)
        else:
            end_sample = end_minute #int(end_minute * fs * 60)

        data=[]
        all_labls = []
        labels = pd.DataFrame(labels_orig)

        indices = np.where(np.array(labels.orig_label) == '+')[0]

        for ind, rpeak_ind in enumerate(labels.index[indices]):

            current_ep = labels.loc[rpeak_ind].ep_label
            rPeak_start = rpeak_ind
            
            if len(indices) > ind+1:
                rPeak_end = labels.index[indices[ind+1]]
            else:
                rPeak_end = len(signal) - 1

            for ind, label in labels.loc[rPeak_start:rPeak_end].iterrows():#range(rPeak_start,rPeak_end):
                #label = labels.loc[ind]

                if label.ep_label == '':
                    label.ep_label = current_ep

        for window_start in range(start_sample, end_sample, self.input_size):
            
            window_end = window_start+self.input_size
            start_ind = np.argmax(labels_orig.index >= window_start)

            '''
            problematic end ind
            
            '''

            end_ind = np.argmax(labels_orig.index >= window_end)

            # print("window: "+str(window_start)+"   "+str(window_end))
            # print("index: "+str(start_ind)+"   "+str(end_ind))

            if start_ind == end_ind:
                if start_ind==0:
                    continue
                else:
                    temp_arr=labels_orig.index <= window_start
                    start_ind = len(labels_orig.index) - np.argmax(temp_arr[::-1]) - 1
                    episode_labels = labels.iloc[start_ind:start_ind+1].copy(deep=True)
                    episode_labels['ind'] = window_start
                    episode_labels = episode_labels.set_index('ind')
            else:
                episode_labels=labels.iloc[start_ind:end_ind]

                # if start_ind!=0:
                #     end_ind = len(signal) # ova e ako nema vekje labels? ne sum sigurna bas koja uloga ja ima
                #     # kako i da e, vo ovoj slucaj na rhythm na afdb ne raboti
                # else:
                #     continue
            #episode_labels=labels.iloc[start_ind:end_ind]
            #print(episode_labels)
            #print(labels.iloc[start_ind:end_ind])
            
            sig = signal[window_start:window_end]

            label = get_episode_label(episode_labels, window_end)

            # plt.plot(sig)
            # plt.suptitle("blabla"+str(label)+"blabla")
            # #plt.suptitle(str(episode_labels))
            # plt.show()

            if label == '':
                continue
            
            #print(label)


            if np.std(sig)==0:
                print("this happened")
                continue
            #data.append(sig)
            data.append(normalize(sig))
            all_labls.append(int(label)) 


        return data, all_labls

    def process(self, X, labels=None):
        # input size is the length of a beat/episode in samples
        # labels is encoded index

        # Basically: get all recordings as X (1 row = 1 30 minute signal)
        # Additional argument: lables but in another format, with R peak locations as index
        # Return segmented episodes, with episode-level labels

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
            print("patient ind"+str(ind))
            full_data.extend(beats)
            full_labels.extend(labls)
            self.groupmap.extend([ind]*len(beats))
            #print(full_labels)

        lengths = []
        for episode in full_data:
            lengths.append(len(episode)/ self.fs)

        # uncommnet for ep label distribution

        #ax = sns.histplot(full_labels, stat='probability')
        #ax.set_title("Episode label distribution for "+self.name+" Dataset")
        #plt.show()
        #for container in ax.containers:
        #    ax.bar_label(container)
        #plt.savefig("./"+self.name+"_ep_distr_mlb.png", dpi=300)

        self.idmap = np.arange(len(full_data))
        print("after processing")
        print(len(full_data))
        print(len(full_labels))
        print(full_labels[0]) 
        return full_data, full_labels # or maybe groupmap?
