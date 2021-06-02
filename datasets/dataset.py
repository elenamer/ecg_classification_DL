
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
# modes: test (everything in one), train-val-test, crossval (2 or 3 parts)

results_path = "./data_overviews"

class Dataset():

    def __init__(self):
        #super(Dataset, self).__init__()
        self.all_morph_classes = []
        self.all_rhy_classes = []

        self.rhythmic_classes = self.get_rhythmic_classes()
        self.morphological_classes = self.get_morphological_classes()




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
        print(self.all_morph_classes)
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
        #print(rhy_dict)
        print(self.all_morph_classes)
        return rhy_dict

    #implement get_class_distributions

    def data_distribution_tables(self):
        results_df_lab, results_df_rhy = self.get_class_distributions()

        print(results_df_lab)
        print(results_df_rhy)
        morph_class_ids = list(set(self.morphological_classes.values()))
        rhy_class_ids = list(set(self.rhythmic_classes.values()))
        #print(results_df_lab)
        print(self.all_morph_classes)
        classes_morph = pd.DataFrame(np.zeros((1, len(self.all_morph_classes))),columns=[str(i) for i in range(len(self.all_morph_classes))]) #results_df_lab.loc['all',:] 
        classes_rhy = pd.DataFrame(np.zeros((1, len(self.all_rhy_classes))),columns=[str(i) for i in range(len(self.all_rhy_classes))]) #results_df_lab.loc['all',:] 

        for key in self.rhythmic_classes.keys():
            if str(key) in results_df_rhy.index:
                print(key)
                print(results_df_rhy.index)
                print(self.rhythmic_classes[key])
                print(classes_rhy.columns)
                classes_rhy[[str(self.rhythmic_classes[str(key)])]]+=results_df_rhy.loc[str(key)]

        for key in self.morphological_classes.keys():
            if str(key) in results_df_lab.index:
                classes_morph[[str(self.morphological_classes[key])]]+=results_df_lab.loc[key]
        
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