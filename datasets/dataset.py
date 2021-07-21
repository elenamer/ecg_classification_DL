
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification


# modes: test (everything in one), train-val-test, crossval (2 or 3 parts)

results_path = "./data_overviews"

class Dataset():

    def __init__(self, task):
        #super(Dataset, self).__init__()
        self.all_morph_classes = []
        self.all_rhy_classes = []
        self.task = task

        self.classes = self.get_classes()
        self.k_fold = IterativeStratification(n_splits=10, order=1) # fixed for now, should be defined by evaluation paradigm in the future


    '''def get_patientids():

    def get_signal()

    def get_annotation()

    def examine_dataset()'''

    def get_classes(self):

        class_mappings = pd.read_csv("tasks/class_mappings.csv", index_col = 0)
        classes_df = pd.read_csv("tasks/"+self.task+"-task.csv", index_col = None)
        print("Classes df")
        print(classes_df)

        #print(classes_df.loc[self.name])
        class_dict = {}
        self.class_names = classes_df.columns

        for i, col in enumerate(classes_df.columns):
            #print(classes_df.loc[self.name][col])
            subclasses =  classes_df[col].values[0]
            print("subclasses")
            print(subclasses)
            for cl in subclasses.split(","):
                #print(class_mappings.loc[self.name])
                cl_nospace = cl.strip()
                if str(class_mappings.loc[self.name][cl_nospace]).lower() != 'nan': 
                    for label in class_mappings.loc[self.name][cl_nospace].split(","):
                        print(class_mappings.loc[self.name][cl_nospace].split(","))
                        class_dict[label.strip()] = i
        #print(rhy_dict)
        print(self.class_names)
        print(class_dict)
        return class_dict

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

    def encode_labels(self):
        mlb_rhy = MultiLabelBinarizer(classes=range(len(self.class_names.values)))
        encoded_index = self.index.copy()
        #encoded_index.set_index("FileName", inplace=True)
        encoded_index["labels_mlb"] = ""
        for ind in encoded_index.index:
            #print(row)
            labls = self.get_annotation(self.path, ind)
            encoded_index.at[ind, "labels_mlb"] = tuple(labls)

        encoded_index["labels_mlb"] = mlb_rhy.fit_transform(encoded_index["labels_mlb"]).tolist()
        print(encoded_index["labels_mlb"])
        encoded_index = encoded_index[["labels_mlb"]]
        print(encoded_index)
        return encoded_index

    def multilabel_distribution_tables(self):
        counts = self.encoded_labels.labels_mlb.apply(lambda x:sum(x))
        print(counts)
        ax = sns.histplot(x=counts)
        ax.set_title(self.task+" multi-label distribution for "+self.name+" Dataset")
        plt.show() 

    # def get_class_distributions(self):
    #     # load and convert annotation data

    #     # overriden in physionetdataset
    #     Y = self.index
    #     count=0
    #     beat_labels = []
    #     rhythm_labels=[]

    #     beat_label_counts = []
    #     rhythm_label_counts = []

    #     for idx in Y.index:
    #         labls, rhythms = self.get_annotation(self.path, idx)
    #         if len(rhythms)>0 and rhythms[0] is not None:
    #             rhythm_labels+=rhythms
    #             rhythm_label_counts.append(len(rhythms)) # always 1 rhythm
    #         beat_labels+=labls
    #         beat_label_counts.append(len(labls))
                
    #     unique_beats = Counter(beat_labels)
    #     unique_rhythm = Counter(rhythm_labels)
    #     results_df_lab = pd.DataFrame.from_dict({"all":unique_beats}, orient='index')
    #     results_df_rhy = pd.DataFrame.from_dict({"all":unique_rhythm}, orient='index')
    #     print("Second")
    #     print(results_df_rhy)
    #     print(results_df_lab)
    #     return results_df_lab.loc["all",:], results_df_rhy.loc["all",:]

    def get_class_distributions(self, list_classes):
       # overriden in physionetdataset
        print(list_classes.shape)
        Y = self.index

        unique_rhythm = {}

        for idx in range(list_classes.shape[1]):       
            unique_rhythm[idx] = sum(list_classes[:,idx])

        results_df_rhy = pd.DataFrame.from_dict({"all":unique_rhythm}, orient='index')
        print("first")
        print(results_df_rhy)
        return results_df_rhy.loc["all",:]

    def data_distribution_tables(self, list_classes=None):

        # load and convert annotation data
        if list_classes is None:
            try:
                list_classes = np.array(self.encoded_labels.labels_mlb.values.tolist())
            except AttributeError:
                list_classes = None

        results_df = self.get_class_distributions(list_classes)

        classes = pd.DataFrame(np.zeros((1, len(self.class_names))),columns=[str(i) for i in range(len(self.class_names))]) #results_df_lab.loc['all',:] 
        print(self.classes)
        for key in self.classes.keys():
            print(key)
            print(self.classes[str(key)])
            if self.classes[str(key)] in results_df.index:
                classes[[str(self.classes[str(key)])]]+=results_df.loc[int(self.classes[str(key)])]


        print(classes)

        classes.to_csv(results_path+os.sep+self.name+"_"+self.task+"_distribution.csv")

        #ax = sns.barplot(x = self.class_names,y = classes.values.flatten())
        #ax.set_title(self.task+" class distribution for "+self.name+" Dataset")
        #plt.show()

        '''
        TEMPORARY: Not needed
        
        '''
        # self.multilabel_distribution_tables()
        
        
        classes.columns = self.class_names
        classes.index = [self.name]
        return classes
