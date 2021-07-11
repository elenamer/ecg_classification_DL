


from warnings import resetwarnings
from models.model import Classifier
import os
import sys
import tensorflow as tf
import numpy as np
import sklearn
import wandb

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
    def __init__(self, dataset, transform, input_seconds, model, task, evaluation_strategy, epochs, save_model = False):
        self.dataset = dataset()
        fs = self.dataset.freq
        self.input_size = int(input_seconds*fs)
        self.transform = transform(self.input_size) # connected with input_size
        self.model = model(dropout=0.1)
        self.task = task
        if task == "rhythm":
            self.classes = self.dataset.all_rhy_classes
        else:
            self.classes = self.dataset.all_morph_classes
        self.eval = evaluation_strategy

        self.path = "experiments"+os.sep+self.dataset.name+os.sep+self.transform.name+str(self.input_size)+os.sep+self.model.model_name+os.sep+self.task  
        os.makedirs(self.path, exist_ok=True)
        self.save = save_model
        self.name = self.dataset.name+"_"+self.transform.name+str(self.input_size)+"_"+self.model.model_name+"_"+self.task  

        self.epochs = epochs
    
    def run(self):

        ## Here choose between evaluation paradigms according to self.eval
        # For now have only one which is obtained with dat.get_crossval_split()

        for n in range(self.dataset.k_fold.get_n_splits()):
            # (look at ptbxl code, basically go through all models for a specific dataset)
            os.makedirs(self.path+os.sep+str(n)+os.sep+"models", exist_ok=True)
            run = wandb.init(project=self.name, reinit=True)            
            wandb.run.name = "crossval"+str(n)
            wandb.run.save()
            tf.keras.backend.clear_session()
            self.classifier = Classifier(self.model, self.input_size, len(self.classes), self.transform, path=self.path+os.sep+str(n), learning_rate=0.001, epochs = self.epochs)
            self.classifier.add_compile()
            self.classifier.summary()
            print(n)
            X_train, Y_train, X_val, Y_val, X_test, Y_test = self.dataset.get_crossval_splits(split=n, task = self.task)
            
            Y_test.dump(self.path+os.sep+str(n)+os.sep+"Y_test.npy") 
            Y_val.dump(self.path+os.sep+str(n)+os.sep+"Y_val.npy") 
            Y_train.dump(self.path+os.sep+str(n)+os.sep+"Y_train.npy") 

            self.classifier.fit(x=X_train,y=Y_train, validation_data = (X_val, Y_val))

            self.classifier.predict(X_test, Y_test).dump(self.path+os.sep+str(n)+os.sep+"Y_test_pred.npy") 
            self.classifier.predict(X_val, Y_val).dump(self.path+os.sep+str(n)+os.sep+"Y_val_pred.npy") 
            self.classifier.predict(X_train, Y_train).dump(self.path+os.sep+str(n)+os.sep+"Y_train_pred.npy") 

            if self.save:
                os.makedirs(self.path+os.sep+str(n)+os.sep+"model", exist_ok=True)
                self.classifier.save(self.path+os.sep+str(n)+os.sep+"model")

            run.finish()



    def evaluate(self):

        # summarize all runs somehow (look at summarize_results script and ptbxl code)
        # calc metrics somehow?

        results_dict = {}

        metric_names = ["auc", "f1"] # for now only mean acc, mean f1 and mean auc; calc var outside
        
        auc_test_scores = []
        auc_val_scores = []
        f1_test_scores = []
        f1_val_scores = []

        for n in range(self.dataset.k_fold.get_n_splits()):

            y_train = np.load(self.path+os.sep+str(n)+os.sep+'Y_train.npy', allow_pickle=True)
            y_val = np.load(self.path+os.sep+str(n)+os.sep+'Y_val.npy', allow_pickle=True)
            y_test = np.load(self.path+os.sep+str(n)+os.sep+'Y_test.npy', allow_pickle=True)
            

            y_train_pred = np.load(self.path+os.sep+str(n)+os.sep+'Y_train_pred.npy', allow_pickle=True)
            y_val_pred = np.load(self.path+os.sep+str(n)+os.sep+'Y_val_pred.npy', allow_pickle=True)
            y_test_pred = np.load(self.path+os.sep+str(n)+os.sep+'Y_test_pred.npy', allow_pickle=True)
            
            labels = np.argwhere(y_train.sum(axis=0) > 0 )
            print(labels)
            print(np.sum(y_test_pred, axis=0))
            print(np.sum(y_test, axis=0))

            auc_test = sklearn.metrics.roc_auc_score(y_test[:,(labels[:,0])], y_test_pred[:,(labels[:,0])], average='macro')
            auc_val = sklearn.metrics.roc_auc_score(y_val[:,(labels[:,0])], y_val_pred[:,(labels[:,0])], average='macro')

            auc_test_scores.append(auc_test)
            auc_val_scores.append(auc_val)

            f1_val = sklearn.metrics.f1_score(y_test.argmax(axis=1), y_test_pred.argmax(axis=1), average = 'macro')
            f1_test = sklearn.metrics.f1_score(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), average = 'macro')

            f1_test_scores.append(f1_test)
            f1_val_scores.append(f1_val)

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
        print(results_dict)

        original_stdout = sys.stdout
        
        with open(self.path+os.sep+'results.txt', 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(results_dict)
            sys.stdout = original_stdout

        return results_dict
