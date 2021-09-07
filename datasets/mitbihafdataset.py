
from .physionetdataset import PhysionetDataset

random_seed=100

choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

aami_annots_list=['N','L','R','e','j','S','A','a','J','V','E','F','/','f','Q']

class AFDataset(PhysionetDataset):

    def __init__(self, task, fs=None, eval='inter', lead = 'II'): ## classes, segmentation, selected channel
        self.name = "afdb"
        super(AFDataset, self).__init__(self.name, task, eval)

        # self.classes = ["N", "S", "V", "F", "Q"]

        # for patient-specific
                
        self.freq = 360
        if fs is not None:
            self.new_freq = fs
        else:
            self.new_freq = self.freq

        ## Update patient splits

        self.common_patients = [101,106,108,109,112,114,115,116,118,119,122,124,100,103,105,111,113,117,121,123]
        self.specific_patients = [201,203,205,207,208,209,215,220,223,230,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

        # for inter-patient
        self.ds1_patients_train = [106,108,109,112,115,116,118,119,122,124,201,203,205,207,208,209,215,220,230]
        self.ds1_patients_val = [101,114,223] 
        self.ds2_patients = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]


        self.lead = lead 

        self.stringify_patientids()
        self.remove_empty_records()

    def remove_empty_records(self):
        temp = self.patientids
        with open(self.path+"notes.txt") as f:
            lines = f.read().splitlines()
        for line in lines:
            words = line.split("\t")
            if words[1] == "Signals unavailable":
                temp.remove(words[0])
        self.patientids = temp

    def stringify_patientids(self):
        self.common_patients = [str(id) for id in self.common_patients]
        self.specific_patients = [str(id) for id in self.specific_patients]

        # for inter-patient
        self.ds1_patients_train = [str(id) for id in self.ds1_patients_train]
        self.ds1_patients_val = [str(id) for id in self.ds1_patients_val]
        self.ds2_patients = [str(id) for id in self.ds2_patients]
