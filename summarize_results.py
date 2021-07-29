import os
import argparse
import numpy as np
from numpy.lib import stride_tricks
import pandas as pd
import pickle

tasks = ["form", "rhythm"]

classes = list(range(10))

metrics = ["AUC","F1"]
stats = ["mean", "var"]#,"std","min","max"]
sets = ["test","val"]

models=['cpscwinner', 'resnet', 'rtacnn','cnn','wavelet']
models_and_all = models # just list them

models_ind = np.repeat(models_and_all, len(metrics)*len(stats) *len(sets))

metrics_ind = np.repeat(metrics, len(stats) * len(sets)) 
metrics_ind = [str(s) for s in metrics_ind]
metrics_ind = metrics_ind * len(models_and_all)

stats_ind = np.repeat(stats, len(sets))
stats_ind = [str(s) for s in stats_ind]
stats_ind = stats_ind * len(metrics)* len(models_and_all)

sets_ind = sets * len(metrics) * len(models_and_all) * len(stats)


datasets_ind = [] 
segmentations_ind = []

datasets = [d for d in os.listdir("experiments") if os.path.isdir("experiments"+os.sep+d)]

print(datasets)

for dataset in datasets:
    segmentations = [d for d in os.listdir("experiments"+os.sep+dataset) if os.path.isdir("experiments"+os.sep+dataset+os.sep+d)]
    for segm in segmentations:
        datasets_ind.append(dataset)
        segmentations_ind.append(segm)

results = {}
print(datasets_ind)
print(segmentations_ind)

results_df1 = pd.DataFrame(columns = [models_ind, metrics_ind, stats_ind, sets_ind], index=[datasets_ind, segmentations_ind] )
results_df1 = results_df1.fillna('/')
print(results_df1)

for task in tasks:

    results_df1 = pd.DataFrame(columns = [models_ind, metrics_ind, stats_ind, sets_ind], index=[datasets_ind, segmentations_ind] )
    results_df1 = results_df1.fillna('/')

    for dataset in datasets:
        segmentations = [d for d in os.listdir("experiments"+os.sep+dataset) if os.path.isdir("experiments"+os.sep+dataset+os.sep+d)]
        for segm in segmentations:
            models = [d for d in os.listdir("experiments"+os.sep+dataset+os.sep+segm) if os.path.isdir("experiments"+os.sep+dataset+os.sep+segm+os.sep+d)]
            for model in models:
                path = "experiments"+os.sep+dataset+os.sep+segm+os.sep+model+os.sep+task
                print(path)
                try:      
                    with open(path+os.sep+'results.txt', 'r') as f:
                        print(f)
                        temp = f.read()
                        dict = eval(temp)
                        print(dict)
                except:
                    continue
                for metric in metrics:
                    for stat in stats:
                        for dset in sets:
                            print(dict[metric.lower()][dset.lower()])
                            elem = dict[metric.lower()][dset.lower()][stat.lower()]
                            results_df1.loc[(dataset, segm),(model, metric, stat, dset)] = elem
    results[task]= results_df1

for task in tasks:
    results[task].to_csv(os.path.join('.', "summary_"+task+".csv"))


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