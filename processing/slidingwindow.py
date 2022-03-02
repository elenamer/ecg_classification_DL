
from processing.transform import Transform
import numpy as np
import matplotlib.pyplot as plt

class SlidingWindow(Transform):
    def __init__(self, input_size):
        super().__init__(input_size)
        self.name = "slidingwindow"

    def process(self, X, labels=None):
        overlap = 0.5
        print("windowing")
        new_data = []
        new_labels = []
        idmap = []
        for ind, sig in enumerate(X):
            if len(sig) == self.input_size:
                print("no need, already windowed")
                new_data = X
                new_labels = labels
                self.idmap = np.arange(new_data.shape[0])
                #print(self.idmap)
                break
            #print(sig)
            step = int(self.input_size*overlap)
            nrows = ((len(sig)-self.input_size)//step)+1
            #print(nrows)
            if nrows <= 0:
                windows = np.array([sig])
                nrows = 1
            else:
                windows = sig[step*np.arange(nrows)[:,None] + np.arange(self.input_size)]
            #print(windows.shape)

            new_data.extend(windows.tolist())
            if labels is not None:
                new_labels.extend([labels[ind]] * nrows)
            idmap.extend([ind] * nrows)
        self.groupmap = idmap
        
        if labels is None:
            new_data = super(SlidingWindow, self).process(new_data)
            return new_data, idmap
            # just crop/pad if needed in base class
            # convert to numpy array

        new_data, new_labels, t = super(SlidingWindow, self).process(new_data, new_labels)
        return new_data, new_labels, idmap
