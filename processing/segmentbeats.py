#Something like transforms, have it as an argument in dataset and call it

from processing.transform import Transform
import numpy as np


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



class SegmentBeats(Transform):

    def __init__(self, input_size):
        self.idmap = [] 
        self.input_size = input_size

    def reset_idmap(self):
        self.idmap = []

    def aggregate_labels(self, preds):

        return preds

    def segment_beats(self, choice, signal, labels, beat_len, start_minute, end_minute):
        # here start minute is basically start sample

        N_SAMPLES_BEFORE_R_static=int(beat_len/2)
        N_SAMPLES_AFTER_R_static=int(beat_len/2)

        # ? I don't know for now
        # N_SAMPLES_BEFORE_R_dynamic=int(fs/4.5)
        print(labels.index)

        print(N_SAMPLES_BEFORE_R_static)

        start_sample = start_minute #int(start_minute * fs * 60)
        start_ind = np.argmax(labels.index >= start_sample)
        if end_minute == -1:
            end_ind = len(signal)
        else: 
            end_sample = end_minute #int(end_minute * fs * 60)
            end_ind = np.argmax(labels.index >= end_sample)

        skipped=0
        next_ind=start_ind
        print("Start index:")
        print(start_ind)
        print("End index:")
        print(end_ind)
        #print(labels)
        data=[]
        all_labls = []

        for ind in labels.index[start_ind:end_ind]:
            rPeak = ind
            #print(labels.loc[ind])
            label=labels.loc[ind]
            #print(label)
            next_ind+=1

            #print(label)
            #print(rPeak)

            if choice=="static":
                if rPeak-N_SAMPLES_BEFORE_R_static <0 or rPeak+N_SAMPLES_AFTER_R_static>len(signal):
                    continue
                sig = signal[rPeak-N_SAMPLES_BEFORE_R_static:rPeak+N_SAMPLES_AFTER_R_static]
                #sig=resample(signal[rPeak-N_SAMPLES_BEFORE_R_static:rPeak+N_SAMPLES_AFTER_R_static], beat_len)

            # else:#if choice=="dynamic": 
            #     if rPeak-N_SAMPLES_BEFORE_R_dynamic <0:
            #         continue        
            #     if len(ann[:,2]) == next_ind:
            #         sig=resample(signal[rPeak-N_SAMPLES_BEFORE_R_dynamic:],beat_len)
            #     else:
            #         rPeak_next=ann[next_ind,1]
            #         print(signal)
            #         print(rPeak-N_SAMPLES_BEFORE_R_dynamic)
            #         print(rPeak_next-N_SAMPLES_BEFORE_R_dynamic)
            #         sig=resample(signal[rPeak-N_SAMPLES_BEFORE_R_dynamic:rPeak_next-N_SAMPLES_BEFORE_R_dynamic],beat_len)
            #     class_label=BEAT_LABEL_TRANSLATIONS[label]
            if np.std(sig)==0:
                print("this happened")
                continue
            data.append(sig)
            # !! data.append(normalize(sig))

            all_labls.append(label)

        # print(skipped)
        # print(len(data))
        # print(len(all_labls))
        # print(len(data[0]))
        # print(all_labls)
        # print("All labels")
        return data, all_labls

    def process(self, X, labels=None, window = False):
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
        choice = "static"
        for ind, sig in enumerate(X):
            if len(sig) == self.input_size:
                print("no need, already segmented")
                full_data = X
                full_labels = labels
                break
            beats, labls = self.segment_beats(choice, sig, labels[ind], self.input_size, 0, 60*360)
            full_data.extend(beats)
            full_labels.extend(labls)
            #print(full_labels)
        full_data, full_labels = super(SegmentBeats, self).process(full_data, full_labels)
        self.idmap = np.arange(full_data.shape[0])
        print("after processing")
        print(full_data.shape)
        print(full_labels.shape)
        return full_data, full_labels