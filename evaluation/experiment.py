


from datasets.physionetdataset import PhysionetDataset
from processing.segmentepisodesthreshold import SegmentEpisodesThreshold
from processing.segmentepisodes import SegmentEpisodes
from warnings import resetwarnings
from models.model import Classifier
import os
import sys
import tensorflow as tf
import numpy as np
import sklearn
import wandb
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize
import gc

def evaluate_metrics(confusion_matrix):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    # Sensitivity, hit rate, recall, or true positive rate
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    support = confusion_matrix.sum(axis=1)
    eps = 0.0000001
    TPR = TP / (TP + FN + eps)
    # Specificity or true negative rate
    TNR = TN / (TN + FP + eps)
    # Precision or positive predictive value
    PPV = TP / (TP + FP + eps)
    # Negative predictive value
    NPV = TN / (TN + FN + eps)
    # Fall out or false positive rate
    FPR = FP / (FP + TN + eps)
    # False negative rate
    FNR = FN / (TP + FN + eps)
    # False discovery rate
    FDR = FP / (TP + FP + eps)
    # F1 score
    F1 = 2 * PPV * TPR / (PPV + TPR + eps)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))

    ACC_macro = np.mean(
        ACC)  # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)
    results_dict = {"ACC_macro":ACC_macro, "ACC":ACC, "TPR":TPR, "TNR":TNR, "PPV":PPV}
    return ACC, TPR, TNR, PPV, F1, support

wandb_flag = True

class Experiment():
    def __init__(self, dataset, transform, freq, input_seconds, model, task, evaluation_strategy, epochs, aggregate = False, save_model = False, episodes = False, max_episode_seconds = 10):
        self.fs = freq
        self.eval = evaluation_strategy
        self.dataset = dataset(task = task, fs = self.fs, eval = self.eval)
        self.input_size = int(input_seconds*self.fs)
        self.transform = transform(self.input_size) # connected with input_size
        self.is_dnn = True
        self.model = model #(dropout=0.1)
        model_name = model.get_name()
        if model_name == "wavelet":
            self.is_dnn=False

        self.task = task
        
        self.classes = self.dataset.class_names

        self.episodes = episodes
        self.episodestransform = SegmentEpisodesThreshold(int(max_episode_seconds*self.fs), self.fs)

        ## max_episode_seconds is different from input_seconds only when episodes+ sliding window are used in combination
        episodes_indicator = self.episodestransform.name if self.episodes else ''

        self.path = "experiments"+os.sep+self.dataset.name+os.sep+episodes_indicator+self.transform.name+str(self.input_size)+os.sep+model_name+os.sep+self.task+os.sep+self.eval  
        os.makedirs(self.path, exist_ok=True)
        self.save = save_model
        self.name = self.dataset.name+"_"+episodes_indicator+self.transform.name+str(self.input_size)+"_"+model_name+"_"+self.task+"_"+str(self.fs)

        self.epochs = epochs
        self.aggregate = aggregate

    
    def run(self):

        ## Here choose between evaluation paradigms according to self.eval
        # For now have only one which is obtained with dat.get_crossval_split()
        distrs_splits = []
        if isinstance(self.dataset, PhysionetDataset):
            if self.eval != "fixed":
                X, Y = self.dataset.get_data()

                if self.episodes:
                    X, Y = self.episodestransform.process(X = X, labels = Y)
                    print(Y)
                    Y = label_binarize(Y, classes=range(len(self.dataset.class_names.values)))
                    if len(self.dataset.class_names.values) == 2:
                        Y = np.array([[0 if item[0] else 1,item[0]] for item in Y])
                    print("after binarizer")
                    print(Y)
                    if self.eval == "inter":
                        groups = self.episodestransform.groupmap

                if not self.episodes:
                    ### intrapatient with segment beats doesn't work with aggregation
                    X, Y, idmap = self.transform.process(X = X, labels = Y)
                    if self.eval == "inter":
                        groups = self.transform.groupmap

        for n in range(self.dataset.n_splits):
            # (look at ptbxl code, basically go through all models for a specific dataset)
            os.makedirs(self.path+os.sep+str(n)+os.sep+"models", exist_ok=True)
            if wandb_flag:
                run = wandb.init(project=self.name, reinit=True, job_type=self.eval)            
                wandb.run.name = "crossval"+str(n)
                wandb.run.save()
            if self.is_dnn:
                tf.keras.backend.clear_session()
                self.classifier = Classifier(self.model(), self.input_size, len(self.classes), path=self.path+os.sep+str(n), learning_rate=0.0001, epochs = self.epochs) ## lr for good cpsc run is ~0.0001 - 0.001
                self.classifier.add_compile()
                self.classifier.summary()
            else:
                self.classifier = self.model(n_classes=len(self.classes), freq=self.fs, outputfolder=self.path+os.sep+str(n))
            print(n)

            if self.eval == 'fixed':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = self.dataset.get_split_interpatient(split=n)
                
                if self.episodes:
                    X_test, Y_test = self.episodestransform.process(X = X_test, labels = Y_test)
                    X_val, Y_val = self.episodestransform.process(X = X_val, labels = Y_val)
                    X_train, Y_train = self.episodestransform.process(X = X_train, labels = Y_train)
                    Y_test = label_binarize(Y_test, classes=range(len(self.dataset.class_names.values)))
                    Y_val = label_binarize(Y_val, classes=range(len(self.dataset.class_names.values)))
                    Y_train = label_binarize(Y_train, classes=range(len(self.dataset.class_names.values)))
                
                X_test, Y_test, idmap_test = self.transform.process(X = X_test, labels = Y_test)
                X_val, Y_val, idmap_val = self.transform.process(X = X_val, labels = Y_val)
                X_train, Y_train, idmap_train = self.transform.process(X = X_train, labels = Y_train)

            elif self.eval=='inter':
                if isinstance(self.dataset, PhysionetDataset):
                    X_train, Y_train, X_val, Y_val, X_test, Y_test = self.dataset.get_crossval_splits(X=X, Y=Y, recording_groups=groups, split=n)
                else:
                    X_train, Y_train, X_val, Y_val, X_test, Y_test = self.dataset.get_crossval_splits(split=n)
                    X_test, Y_test, idmap_test = self.transform.process(X = X_test, labels = Y_test)
                    X_val, Y_val, idmap_val = self.transform.process(X = X_val, labels = Y_val)
                    X_train, Y_train, idmap_train = self.transform.process(X = X_train, labels = Y_train)
                              

                if self.episodes:
                    X_test, Y_test, idmap_test = self.transform.process(X = X_test, labels = Y_test)
                    X_val, Y_val, idmap_val = self.transform.process(X = X_val, labels = Y_val)
                    X_train, Y_train, idmap_train = self.transform.process(X = X_train, labels = Y_train)
               
            else:

                X_train, Y_train, X_val, Y_val, X_test, Y_test = self.dataset.get_crossval_splits_intrapatient(X=X, Y=Y, split=n)

                if self.episodes:
                    X_test, Y_test, idmap_test = self.transform.process(X = X_test, labels = Y_test)
                    X_val, Y_val, idmap_val = self.transform.process(X = X_val, labels = Y_val)
                    X_train, Y_train, idmap_train = self.transform.process(X = X_train, labels = Y_train)

            X_train = np.asarray(X_train, dtype=np.float16)
            Y_train = np.asarray(Y_train, dtype=np.int8)
            X_val = np.asarray(X_val, dtype=np.float16)
            Y_val = np.asarray(Y_val, dtype=np.int8)
            X_test = np.asarray(X_test, dtype=np.float16)
            Y_test = np.asarray(Y_test, dtype=np.int8)
            if Y_test.shape[1] == 2:
                Y_train_input = np.array([0 if item[0] else 1 for item in Y_train])
                Y_test_input = np.array([0 if item[0] else 1 for item in Y_test])
                Y_val_input = np.array([0 if item[0] else 1 for item in Y_val])
            else:
                Y_train_input = Y_train
                Y_test_input = Y_test
                Y_val_input = Y_val
            print("after processing")
            print("class distribution:")
            print(Y_train.shape)
            print(Y_train[0].shape)
            print(Y_train.sum(axis=0))
            print(Y_test.sum(axis=0))
            print(Y_val.sum(axis=0))
            print(X_train.shape)
            print(X_val.shape)
            print(Y_val.shape)
            print(X_train.dtype)
            print(Y_train.dtype)
            print(X_test.dtype)
            #['N' '' list([1, 0, 0, 0, 0]) '']

            np.save(self.path+os.sep+str(n)+os.sep+"Y_test.npy",Y_test) 
            np.save(self.path+os.sep+str(n)+os.sep+"Y_val.npy",Y_val) 
            np.save(self.path+os.sep+str(n)+os.sep+"Y_train.npy", Y_train) 

            times = self.classifier.fit(x=X_train,y=Y_train_input, validation_data = (X_val, Y_val_input))

            Y_test_pred = self.classifier.predict(X_test)
            np.save(self.path+os.sep+str(n)+os.sep+"Y_test_pred.npy", Y_test_pred) 
            Y_val_pred = self.classifier.predict(X_val)
            np.save(self.path+os.sep+str(n)+os.sep+"Y_val_pred.npy", Y_val_pred) 
            Y_train_pred = self.classifier.predict(X_train)
            np.save(self.path+os.sep+str(n)+os.sep+"Y_train_pred.npy", Y_train_pred) 

            np.save(self.path+os.sep+str(n)+os.sep+"epoch_times.npy", np.array(times)) 

            if self.aggregate:
                Y_train_pred_agg = self.transform.aggregate_labels(Y_train_pred, idmap_train)
                np.save(self.path+os.sep+str(n)+os.sep+"Y_train_pred_agg.npy",Y_train_pred_agg) 
                Y_test_pred_agg = self.transform.aggregate_labels(Y_test_pred, idmap_test)
                np.save(self.path+os.sep+str(n)+os.sep+"Y_test_pred_agg.npy", Y_test_pred_agg) 
                Y_val_pred_agg = self.transform.aggregate_labels(Y_val_pred, idmap_val)
                np.save(self.path+os.sep+str(n)+os.sep+"Y_val_pred_agg.npy", Y_val_pred_agg)     


                np.save(self.path+os.sep+str(n)+os.sep+"Y_train_agg.npy", self.transform.aggregate_labels(Y_train, idmap_train)) 
                np.save(self.path+os.sep+str(n)+os.sep+"Y_val_agg.npy", self.transform.aggregate_labels(Y_val, idmap_val)) 
                np.save(self.path+os.sep+str(n)+os.sep+"Y_test_agg.npy", self.transform.aggregate_labels(Y_test, idmap_test)) 

            if self.save:
                os.makedirs(self.path+os.sep+str(n)+os.sep+"model", exist_ok=True)
                self.classifier.save(self.path+os.sep+str(n)+os.sep+"model")
            if wandb_flag:
                run.finish()
                
            distrs_splits.append(np.sum(Y_test, axis=0))
        
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 14

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        fig = plt.figure(figsize=(17,8))
        axes = fig.add_subplot(1,1,1)
        df = pd.DataFrame(distrs_splits)
        df = df.astype(int)
        eps = 10e-1
        
        sns.heatmap(df+eps, annot=df.values, fmt="d")
        labels = axes.get_yticklabels()
        axes.set_yticklabels(labels, rotation=0) 
        fig.tight_layout()
        fig.savefig(self.path+os.sep+self.task+'-labels-heatmap.png', dpi=400)


    def evaluate(self):

        # summarize all runs somehow (look at summarize_results script and ptbxl code)
        # calc metrics somehow?

        results_dict = {}

        metric_names = ["auc", "f1"] # for now only mean acc, mean f1 and mean auc; calc var outside
        
        auc_test_scores = []
        auc_val_scores = []
        f1_test_scores = []
        f1_val_scores = []
        all_epoch_times = []

        for n in range(self.dataset.n_splits):

            #y_train = np.load(self.path+os.sep+str(n)+os.sep+'Y_train.npy', allow_pickle=True)
            y_val = np.load(self.path+os.sep+str(n)+os.sep+'Y_val.npy', allow_pickle=True)
            y_test = np.load(self.path+os.sep+str(n)+os.sep+'Y_test.npy', allow_pickle=True)
            

            #y_train_pred = np.load(self.path+os.sep+str(n)+os.sep+'Y_train_pred.npy', allow_pickle=True)
            y_val_pred = np.load(self.path+os.sep+str(n)+os.sep+'Y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(self.path+os.sep+str(n)+os.sep+'Y_test_pred.npy', allow_pickle=True)

            epoch_times = np.load(self.path+os.sep+str(n)+os.sep+'epoch_times.npy', allow_pickle=True)
            
            labels_test = np.argwhere(y_test.sum(axis=0) > 0 )
            labels_val = np.argwhere(y_val.sum(axis=0) > 0 )
            print(labels_test[:,0])
            print(np.sum(y_test_pred, axis=0))
            print(np.sum(y_test, axis=0))

            if self.aggregate:

                y_val_agg = np.load(self.path+os.sep+str(n)+os.sep+'Y_val_agg.npy', allow_pickle=True)
                y_test_agg = np.load(self.path+os.sep+str(n)+os.sep+'Y_test_agg.npy', allow_pickle=True)
                

                y_val_pred_agg = np.load(self.path+os.sep+str(n)+os.sep+'Y_val_pred_agg.npy', allow_pickle=True)
                y_test_pred_agg = np.load(self.path+os.sep+str(n)+os.sep+'Y_test_pred_agg.npy', allow_pickle=True)
                print(y_test_agg[:,(labels_test[:,0])])
                print(y_test_pred_agg[:,(labels_test[:,0])])
                auc_test = sklearn.metrics.roc_auc_score(y_test_agg[:,(labels_test[:,0])], y_test_pred_agg[:,(labels_test[:,0])], average='macro')
                auc_val = sklearn.metrics.roc_auc_score(y_val_agg[:,(labels_val[:,0])], y_val_pred_agg[:,(labels_val[:,0])], average='macro')

                auc_test_scores.append(auc_test)
                auc_val_scores.append(auc_val)

                f1_val = sklearn.metrics.f1_score(y_test_agg.argmax(axis=1), y_test_pred_agg.argmax(axis=1), average = 'macro')
                f1_test = sklearn.metrics.f1_score(y_val_agg.argmax(axis=1), y_val_pred_agg.argmax(axis=1), average = 'macro')

                f1_test_scores.append(f1_test)
                f1_val_scores.append(f1_val)
                del y_val_agg
                del y_test_agg
                del y_val_pred_agg
                del y_test_pred_agg

            else:

                auc_test = sklearn.metrics.roc_auc_score(y_test[:,(labels_test[:,0])], y_test_pred[:,(labels_test[:,0])], average='macro')
                auc_val = sklearn.metrics.roc_auc_score(y_val[:,(labels_val[:,0])], y_val_pred[:,(labels_val[:,0])], average='macro')

                auc_test_scores.append(auc_test)
                auc_val_scores.append(auc_val)

                f1_val = sklearn.metrics.f1_score(y_test.argmax(axis=1), y_test_pred.argmax(axis=1), average = 'macro')
                f1_test = sklearn.metrics.f1_score(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), average = 'macro')

                f1_test_scores.append(f1_test)
                f1_val_scores.append(f1_val)
            all_epoch_times.extend(epoch_times)

            del y_val
            del y_test
            del y_val_pred
            del y_test_pred
            del epoch_times
            gc.collect()


        results_dict["auc"] = {"test": { 
            "mean" : np.mean(np.array(auc_test_scores)),
            "var"  : np.var(np.array(auc_test_scores))
        }, 
        "val": { 
            "mean" : np.mean(np.array(auc_val_scores)),
            "var"  : np.var(np.array(auc_val_scores))
        }, 
        }


        results_dict["f1"] = {"test": { 
            "mean" : np.mean(np.array(f1_test_scores)),
            "var"  : np.var(np.array(f1_test_scores))
        }, 
        "val": { 
            "mean" : np.mean(np.array(f1_val_scores)),
            "var"  : np.var(np.array(f1_val_scores))
        }, 
        }
        results_dict["time"] = {"epoch": { 
            "mean" : np.mean(np.array(all_epoch_times)),
            "var"  : np.var(np.array(all_epoch_times))
        }
        }
        print(results_dict)

        original_stdout = sys.stdout
        
        with open(self.path+os.sep+'results.txt', 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(results_dict)
            sys.stdout = original_stdout

        return results_dict
