
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
# modes: test (everything in one), train-val-test, crossval (2 or 3 parts)

results_path = "./data_overviews"

class Dataset():

    def __init__(self):
        #super(Dataset, self).__init__()
        self.all_morph_classes = []
        self.all_rhy_classes = []

        self.rhythmic_classes = self.get_rhythmic_classes()
        self.morphological_classes = self.get_morphological_classes()

    def get_class_distributions(self):
        """ Implement here a function that returns 2 dicts, for morph. and rhy. class distributions"""
        raise NotImplementedError(
            "Please implement the `get_class_distributions` method for your dataset"
        )

    '''def get_patientids():

    def get_ecgwave()

    def examine_dataset()'''

    def get_rhythmic_classes(self):
        classes_df = pd.read_csv("tasks/rhythmic_classes.csv", index_col = 0)
        print("Classes df")
        print(classes_df)

        #print(classes_df.loc[self.name])
        rhy_dict = {}
        self.all_rhy_classes = classes_df.columns

        for i, col in enumerate(classes_df.columns):
            #print(classes_df.loc[self.name][col])
            if str(classes_df.loc[self.name][col]).lower() != 'nan': 
                for cl in classes_df.loc[self.name][col].split(","):
                    rhy_dict[cl.strip()] = i
        #print(rhy_dict)
        print(self.all_rhy_classes)
        print(rhy_dict)
        return rhy_dict
    
    def get_morphological_classes(self):
        classes_df = pd.read_csv("tasks/morphological_classes.csv", index_col = 0)
        print("Classes df")
        print(classes_df)
        #print(classes_df.loc[self.name])
        rhy_dict = {}
        self.all_morph_classes = classes_df.columns

        for i, col in enumerate(classes_df.columns):
            #print(classes_df.loc[self.name][col])
            if str(classes_df.loc[self.name][col]).lower() != 'nan': 
                for cl in classes_df.loc[self.name][col].split(","):
                    rhy_dict[cl.strip()] = i
        print(self.all_morph_classes)
        print(rhy_dict)
        return rhy_dict

    def get_class_distributions(self):
        # load and convert annotation data
        Y = self.index
        count=0
        beat_labels = []
        rhythm_labels=[]

        beat_label_counts = []
        rhythm_label_counts = []

        for idx in Y.index:
            labls, rhythms = self.get_annotation(self.path, idx)
            if len(rhythms)>0 and rhythms[0] is not None:
                rhythm_labels+=rhythms
                rhythm_label_counts.append(len(rhythms)) # always 1 rhythm
            beat_labels+=labls
            beat_label_counts.append(len(labls))
                
        unique_beats = Counter(beat_labels)
        unique_rhythm = Counter(rhythm_labels)
        results_df_lab = pd.DataFrame.from_dict({"all":unique_beats}, orient='index')
        results_df_rhy = pd.DataFrame.from_dict({"all":unique_rhythm}, orient='index')
        print(results_df_rhy)
        print(results_df_lab)
        return results_df_lab.loc["all",:], results_df_rhy.loc["all",:]

    def data_distribution_tables(self):
        results_df_lab, results_df_rhy = self.get_class_distributions()

        print(results_df_lab)
        print(results_df_rhy)
        morph_class_ids = list(set(self.morphological_classes.values()))
        rhy_class_ids = list(set(self.rhythmic_classes.values()))
        classes_morph = pd.DataFrame(np.zeros((1, len(self.all_morph_classes))),columns=[str(i) for i in range(len(self.all_morph_classes))]) #results_df_lab.loc['all',:] 
        classes_rhy = pd.DataFrame(np.zeros((1, len(self.all_rhy_classes))),columns=[str(i) for i in range(len(self.all_rhy_classes))]) #results_df_lab.loc['all',:] 
        print(self.rhythmic_classes)
        for key in self.rhythmic_classes.keys():
            print(key)
            print(self.rhythmic_classes[str(key)])
            if self.rhythmic_classes[str(key)] in results_df_rhy.index:
                classes_rhy[[str(self.rhythmic_classes[str(key)])]]+=results_df_rhy.loc[int(self.rhythmic_classes[str(key)])]
        for key in self.morphological_classes.keys():
            if self.morphological_classes[str(key)] in results_df_lab.index:
                classes_morph[[str(self.morphological_classes[str(key)])]]+=results_df_lab.loc[int(self.morphological_classes[str(key)])]
        
        print(classes_morph)
        print(classes_rhy)
        classes_morph.to_csv(results_path+os.sep+self.name+"_morphological_distribution.csv")
        classes_rhy.to_csv(results_path+os.sep+self.name+"_rhythmic_distribution.csv")
        ax = sns.barplot(x = self.all_rhy_classes,y = classes_rhy.values.flatten())
        ax.set_title("Rhythmic class distribution for "+self.name+" Dataset")
        plt.show()
        ax = sns.barplot(x = self.all_morph_classes,y = classes_morph.values.flatten())        
        ax.set_title("Morphological class distribution for "+self.name+" Dataset")
        plt.show()

        # ax = sns.histplot(x=beat_counts)
        # ax.set_title("Morphological multi-label distribution for "+self.name+" Dataset")
        # plt.show()

        # ax = sns.histplot(x=rhythm_counts)
        # ax.set_title("Rhythmic multi-label distribution for "+self.name+" Dataset")
        # plt.show()
