from .physionetdataset import PhysionetDataset

random_seed=100

choices = ['_train','_test']
choice = ''
results_path = "./data_overviews"

aami_annots_list=['N','L','R','e','j','S','A','a','J','V','E','F','/','f','Q']

class MITBIHARDataset(PhysionetDataset):

    def __init__(self): ## classes, segmentation, selected channel
        name = 'mitdb'
        self.name = name
        super(MITBIHARDataset, self).__init__(name)

        self.classes = ["N", "S", "V", "F", "Q"]
        self.freq = 360

        # for patient-specific
        self.common_patients = [101,106,108,109,112,114,115,116,118,119,122,124,100,103,105,111,113,117,121,123]
        self.specific_patients = [201,203,205,207,208,209,215,220,223,230,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

        # for inter-patient
        self.ds1_patients_train = [106,108,109,112,115,116,118,119,122,124,201,203,205,207,208,209,215,220,230]
        self.ds1_patients_val = [101,114,223] 
        self.ds2_patients = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]


        self.stringify_patientids()


    def stringify_patientids(self):
        self.common_patients = [str(id) for id in self.common_patients]
        self.specific_patients = [str(id) for id in self.specific_patients]

        # for inter-patient
        self.ds1_patients_train = [str(id) for id in self.ds1_patients_train]
        self.ds1_patients_val = [str(id) for id in self.ds1_patients_val]
        self.ds2_patients = [str(id) for id in self.ds2_patients]

