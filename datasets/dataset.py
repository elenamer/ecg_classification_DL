

import pandas as pd
# modes: test (everything in one), train-val-test, crossval (2 or 3 parts)

class Dataset():

    def __init__(self):
        #super(Dataset, self).__init__()

        self.rhythmic_classes = self.get_rhythmic_classes()
        self.morphological_classes = self.get_morphological_classes()

    '''def get_patientids():

    def get_ecgwave()

    def examine_dataset()'''

    def get_rhythmic_classes(self):
        classes_df = pd.read_csv("tasks/rhythmic_classes.csv", index_col = 0)
        print(classes_df.loc[self.name])
        rhy_dict = {}
        for i, col in enumerate(classes_df.columns):
            print(classes_df.loc[self.name][col])
            if str(classes_df.loc[self.name][col]).lower() != 'nan': 
                for cl in classes_df.loc[self.name][col].split(","):
                    rhy_dict[cl.strip()] = i
        print(rhy_dict)
        return rhy_dict
    
    def get_morphological_classes(self):
        classes_df = pd.read_csv("tasks/morphological_classes.csv", index_col = 0)
        print(classes_df.loc[self.name])
        rhy_dict = {}

        for i, col in enumerate(classes_df.columns):
            print(classes_df.loc[self.name][col])
            if str(classes_df.loc[self.name][col]).lower() != 'nan': 
                for cl in classes_df.loc[self.name][col].split(","):
                    rhy_dict[cl.strip()] = i
        print(rhy_dict)
        return rhy_dict


