
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

from .physionetdataset import PhysionetDataset

random_seed=100

choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

aami_annots_list=['N','L','R','e','j','S','A','a','J','V','E','F','/','f','Q']


class SavvyDataset(PhysionetDataset):
    def __init__(self, name): ## classes, segmentation, selected channel
        super(SavvyDataset, self).__init__(name)

        self.classes = ["N", "S", "V", "F", "Q"]
        self.common_path = "./data/mitdb/"
        print(self.common_path)
        # for patient-specific
        self.common_patients = [101,106,108,109,112,114,115,116,118,119,122,124,100,103,105,111,113,117,121,123]
        self.specific_patients = [4, 21, 31, 32]
        self.stringify_patientids()

    
    def stringify_patientids(self):
        self.common_patients = [str(id) for id in self.common_patients]
        self.specific_patients = [str(id) for id in self.specific_patients]
