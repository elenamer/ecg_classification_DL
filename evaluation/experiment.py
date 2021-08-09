


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



class Experiment():
    def __init__(self, dataset, transform, freq, input_seconds, model, task, evaluation_strategy, epochs, aggregate = False, save_model = False):
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

        self.path = "experiments"+os.sep+self.dataset.name+os.sep+self.transform.name+str(self.input_size)+os.sep+model_name+os.sep+self.task  
        os.makedirs(self.path, exist_ok=True)
        self.save = save_model
        self.name = self.dataset.name+"_"+self.transform.name+str(self.input_size)+"_"+model_name+"_"+self.task+"_"+str(self.fs)

        self.epochs = epochs
        self.aggregate = aggregate
    
    def run(self):

        ## Here choose between evaluation paradigms according to self.eval
        # For now have only one which is obtained with dat.get_crossval_split()
        distrs_splits = []
        for n in range(self.dataset.k_fold.get_n_splits()):
            # (look at ptbxl code, basically go through all models for a specific dataset)
            os.makedirs(self.path+os.sep+str(n)+os.sep+"models", exist_ok=True)
            run = wandb.init(project=self.name, reinit=True)            
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

            if self.eval == 'inter':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = self.dataset.get_crossval_splits(split=n)
                
                self.transform.reset_idmap()

                X_test, Y_test, idmap_test = self.transform.process(X = X_test, labels = Y_test)
                X_val, Y_val, idmap_val = self.transform.process(X = X_val, labels = Y_val)
                X_train, Y_train, idmap_train = self.transform.process(X = X_train, labels = Y_train)

            else:
                
                X, Y = self.dataset.get_data()
                self.transform.reset_idmap()

                X, Y, idmap = self.transform.process(X = X, labels = Y)

                X_train, Y_train, X_val, Y_val, X_test, Y_test = self.dataset.get_crossval_splits_intrapatient(X=X, Y=Y, split=n)


            print("after processing")


            Y_test.dump(self.path+os.sep+str(n)+os.sep+"Y_test.npy") 
            Y_val.dump(self.path+os.sep+str(n)+os.sep+"Y_val.npy") 
            Y_train.dump(self.path+os.sep+str(n)+os.sep+"Y_train.npy") 

            times = self.classifier.fit(x=X_train,y=Y_train, validation_data = (X_val, Y_val))

            Y_test_pred = self.classifier.predict(X_test)
            Y_test_pred.dump(self.path+os.sep+str(n)+os.sep+"Y_test_pred.npy") 
            Y_val_pred = self.classifier.predict(X_val)
            Y_val_pred.dump(self.path+os.sep+str(n)+os.sep+"Y_val_pred.npy") 
            Y_train_pred = self.classifier.predict(X_train)
            Y_train_pred.dump(self.path+os.sep+str(n)+os.sep+"Y_train_pred.npy") 

            np.array(times).dump(self.path+os.sep+str(n)+os.sep+"epoch_times.npy") 

            if self.aggregate:
                Y_train_pred_agg = self.transform.aggregate_labels(Y_train_pred, idmap_train)
                Y_train_pred_agg.dump(self.path+os.sep+str(n)+os.sep+"Y_train_pred_agg.npy") 
                Y_test_pred_agg = self.transform.aggregate_labels(Y_test_pred, idmap_test)
                Y_test_pred_agg.dump(self.path+os.sep+str(n)+os.sep+"Y_test_pred_agg.npy") 
                Y_val_pred_agg = self.transform.aggregate_labels(Y_val_pred, idmap_val)
                Y_val_pred_agg.dump(self.path+os.sep+str(n)+os.sep+"Y_val_pred_agg.npy")     


                self.transform.aggregate_labels(Y_train, idmap_train).dump(self.path+os.sep+str(n)+os.sep+"Y_train_agg.npy") 
                self.transform.aggregate_labels(Y_val, idmap_val).dump(self.path+os.sep+str(n)+os.sep+"Y_val_agg.npy") 
                self.transform.aggregate_labels(Y_test, idmap_test).dump(self.path+os.sep+str(n)+os.sep+"Y_test_agg.npy") 

            if self.save:
                os.makedirs(self.path+os.sep+str(n)+os.sep+"model", exist_ok=True)
                self.classifier.save(self.path+os.sep+str(n)+os.sep+"model")
            run.finish()
        #     distrs_splits.append(np.sum(Y_test, axis=0))
        
        # SMALL_SIZE = 8
        # MEDIUM_SIZE = 10
        # BIGGER_SIZE = 14

        # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        # plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        # plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        # plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        # #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        # #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # fig = plt.figure(figsize=(17,8))
        # axes = fig.add_subplot(1,1,1)
        # df = pd.DataFrame(distrs_splits)
        # df = df.astype(int)
        # eps = 10e-1
        
        # sns.heatmap(df+eps, annot=df.values, fmt="d")
        # labels = axes.get_yticklabels()
        # axes.set_yticklabels(labels, rotation=0) 
        # fig.tight_layout()
        # fig.savefig(self.path+os.sep+self.task+'-labels-heatmap.png', dpi=400)


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

        for n in range(2):

            y_train = np.load(self.path+os.sep+str(n)+os.sep+'Y_train.npy', allow_pickle=True)
            y_val = np.load(self.path+os.sep+str(n)+os.sep+'Y_val.npy', allow_pickle=True)
            y_test = np.load(self.path+os.sep+str(n)+os.sep+'Y_test.npy', allow_pickle=True)
            

            y_train_pred = np.load(self.path+os.sep+str(n)+os.sep+'Y_train_pred.npy', allow_pickle=True)
            y_val_pred = np.load(self.path+os.sep+str(n)+os.sep+'Y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(self.path+os.sep+str(n)+os.sep+'Y_test_pred.npy', allow_pickle=True)

            epoch_times = np.load(self.path+os.sep+str(n)+os.sep+'epoch_times.npy', allow_pickle=True)

            labels = np.argwhere(y_train.sum(axis=0) > 30 )
      
            print(labels)
            print(np.sum(y_test_pred, axis=0))
            print(np.sum(y_test, axis=0))

            if self.aggregate:

                y_val_agg = np.load(self.path+os.sep+str(n)+os.sep+'Y_val_agg.npy', allow_pickle=True)
                y_test_agg = np.load(self.path+os.sep+str(n)+os.sep+'Y_test_agg.npy', allow_pickle=True)
                

                y_val_pred_agg = np.load(self.path+os.sep+str(n)+os.sep+'Y_val_pred_agg.npy', allow_pickle=True)
                y_test_pred_agg = np.load(self.path+os.sep+str(n)+os.sep+'Y_test_pred_agg.npy', allow_pickle=True)
                
                auc_test = sklearn.metrics.roc_auc_score(y_test_agg[:,(labels[:,0])], y_test_pred_agg[:,(labels[:,0])], average='macro')
                auc_val = sklearn.metrics.roc_auc_score(y_val_agg[:,(labels[:,0])], y_val_pred_agg[:,(labels[:,0])], average='macro')

                auc_test_scores.append(auc_test)
                auc_val_scores.append(auc_val)

                f1_val = sklearn.metrics.f1_score(y_test_agg.argmax(axis=1), y_test_pred_agg.argmax(axis=1), average = 'macro')
                f1_test = sklearn.metrics.f1_score(y_val_agg.argmax(axis=1), y_val_pred_agg.argmax(axis=1), average = 'macro')

                f1_test_scores.append(f1_test)
                f1_val_scores.append(f1_val)

            else:

                auc_test = sklearn.metrics.roc_auc_score(y_test[:,(labels[:,0])], y_test_pred[:,(labels[:,0])], average='macro')
                auc_val = sklearn.metrics.roc_auc_score(y_val[:,(labels[:,0])], y_val_pred[:,(labels[:,0])], average='macro')

                auc_test_scores.append(auc_test)
                auc_val_scores.append(auc_val)

                f1_val = sklearn.metrics.f1_score(y_test.argmax(axis=1), y_test_pred.argmax(axis=1), average = 'macro')
                f1_test = sklearn.metrics.f1_score(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), average = 'macro')

                f1_test_scores.append(f1_test)
                f1_val_scores.append(f1_val)
            all_epoch_times.extend(epoch_times)

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
