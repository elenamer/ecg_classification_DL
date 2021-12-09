import os
import argparse
import math
import numpy as np
from numpy.lib import stride_tricks
import pandas as pd
import pickle

root = "experiments-marvin"
tasks = ["aami"]


metrics = ["AUC","F1"]
stats = ["mean"]#,"std","min","max"]
sets = ["test"]#,"val"]
evals = {'intra':['',os.sep+'intra'],
'inter':[os.sep+'inter']
}

models=  [d for d in os.listdir(root) if os.path.isdir(root+os.sep+d)] #['cpscwinner', 'resnet', 'rtacnn','cnn']
models_and_all = models # just list them
stats= list(evals.keys())

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

datasets = [d for d in os.listdir(root) if os.path.isdir(root+os.sep+d)]

print(datasets)

for dataset in datasets:
    segmentations = [d for d in os.listdir(root+os.sep+dataset) if os.path.isdir(root+os.sep+dataset+os.sep+d)]
    for segm in segmentations:
        current_models = [m for m in os.listdir(root+os.sep+dataset+os.sep+segm) if os.path.isdir(root+os.sep+dataset+os.sep+segm+os.sep+m)]
        for mod in current_models:
            datasets_ind.append(mod)
            segmentations_ind.append(segm)

results = {}
print(datasets_ind)
print(segmentations_ind)
print(stats_ind)
results_df1 = pd.DataFrame(columns = [models_ind, metrics_ind, stats_ind, sets_ind], index=[datasets_ind, segmentations_ind] )
results_df1 = results_df1.fillna('/')
print(results_df1)

for task in tasks:

    print(task)

    results_df1 = pd.DataFrame(columns = [models_ind, metrics_ind, stats_ind, sets_ind], index=[datasets_ind, segmentations_ind] )
    results_df1 = results_df1.fillna('/')
    print(results_df1.index.names)
    results_df1 = results_df1[~results_df1.index.duplicated(keep='first')]
    for eval_p in evals.keys():
        for dataset in datasets:
            segmentations = [d for d in os.listdir(root+os.sep+dataset) if os.path.isdir(root+os.sep+dataset+os.sep+d)]
            for segm in segmentations:
                models = [d for d in os.listdir(root+os.sep+dataset+os.sep+segm) if os.path.isdir(root+os.sep+dataset+os.sep+segm+os.sep+d)]
                for model in models:
                    for opt in evals[eval_p]:
                        path = root+os.sep+dataset+os.sep+segm+os.sep+model+os.sep+task+opt
                        print(path)
                        try:      
                            with open(path+os.sep+'results.txt', 'r') as f:
                                #print(f)
                                temp = f.read()
                                #print(temp)
                                dict = eval(temp)
                                #print(dict)
                        except:
                            continue
                        for metric in metrics:
                            for dset in sets:
                                #for stat in stats:
                                print(dict[metric.lower()][dset.lower()])
                                elem = str(round(dict[metric.lower()][dset.lower()]['mean'], 4))+" ("+str(round(math.sqrt(dict[metric.lower()][dset.lower()]['var']), 4))+")"
                                results_df1.loc[(model, segm),(dataset, metric, eval_p, dset)] = elem
        results[(task, eval_p)]= results_df1
print(results[(task, eval_p)])
for task in tasks:
    for eval_p in evals.keys():
        print(task, eval_p)
        new_index= []
        for row in results[(task, eval_p)].itertuples():
            for el in row[1:]:
                if el != '/':
                    new_index.append(row[0])
                    print(row[0])
                    break
        print(len(new_index))
        print(new_index)
        print(results[(task, eval_p)])
        print(results[(task, eval_p)].loc[('cpscwinner', 'segmentepisodesthresholdstandard5000')])
        results[(task, eval_p)].loc[new_index].sort_index(level=0).to_csv(os.path.join('.', "summary_"+task+"_"+eval_p+".csv"))


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