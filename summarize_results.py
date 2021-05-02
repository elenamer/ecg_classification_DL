import os
import argparse
import numpy as np
import pandas as pd
import pickle

'''

ACC
PPV - Precision
TPR - Sen - Recall 
TNR - Spe
F1 

'''

parser = argparse.ArgumentParser(description='Choose root path of all result files')
parser.add_argument('--path', type=str,
                    help='path to results')
args = parser.parse_args()
results_path = args.path

classes = ["N","S","V","F","Q"]

metrics = ["ACC","TNR", "PPV","TPR","F1","support"]
experiments = ["specific", "intra","inter"]
patients = {"specific":[201,203,205,207,208,209,215,220,223,230,200,202,210,212,213,214,219,221,222,228,231,232,233,234],
"inter": [''],
"intra": [0,1,2,3,4,5,6,7,8,9]
}
segmentations = ["static", "dynamic"]#,"incartdb","svdb","edb"]#,"incartdb","svdb","edb"]#,'birnn'] # "birnn", "deeptransf"
models = sorted(os.listdir(results_path))

models = [
"test",
"val"
]

print(models)

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
    results_dict = {"ACC_macro":ACC_macro, "ACC":ACC, "TPR":TPR, "TNR":TNR, "PPV":PPV }
    return ACC, TPR, TNR, PPV, F1, support

'''
for root, dirs, files in os.walk(results_path, topdown=False):
   for name in files:
       if name.endswith(".csv") and "test" not in name:
            print(name)
            try:
                arr = np.loadtxt(os.path.join(root, name))
                # if it is only numpy then it hasn't been processed yet
            except:
                # if numpy load fails, then it has been processed and do nothing
                continue
            print(arr)
            print(os.path.join(root, name))
            if "ACC_macro" in name:
                df = pd.DataFrame(arr, columns=["average"], index=np.arange(1,arr.shape[0]+1))
            else:
                df = pd.DataFrame(arr, columns=classes, index=np.arange(1,arr.shape[0]+1))
            df.loc["mean"]=df.mean()
            df.loc["std"]=df.std()
            print(df)
            df.to_csv(os.path.join(root, name), float_format="%.5f")
'''

segmentations_ind = segmentations*len(models)
models_ind = np.repeat(models, len(segmentations))
stats = ["mean"]#,"std","min","max"]

classes_and_all = ["All"] + classes
classes_ind = np.repeat(classes_and_all, len(metrics)*len(stats))
metrics_ind = np.repeat(metrics, len(stats)) 
metrics_ind = [str(s) for s in metrics_ind]
metrics_ind = metrics_ind * len(classes_and_all)
stats_ind = stats * len(metrics) * len(classes_and_all)
#
print(len(stats_ind))
print(len(classes_ind))
print(len(metrics_ind))
results_df1 = pd.DataFrame(columns = [classes_ind, metrics_ind, stats_ind], index=[models_ind, segmentations_ind] )
results_df1 = results_df1.fillna('/')
results_df2 = pd.DataFrame(columns = [classes_ind, metrics_ind, stats_ind], index=[models_ind, segmentations_ind] )
results_df2 = results_df2.fillna('/')
results_df3 = pd.DataFrame(columns = [classes_ind, metrics_ind, stats_ind], index=[models_ind, segmentations_ind] )
results_df3 = results_df2.fillna('/')
results = {"intra": results_df1, "inter": results_df2, "specific": results_df3}

print("BREAK")
for model in models:
    for segmentation_method in segmentations:
        for evaluation_method in experiments:
            skip=False
            ACC=[]
            TPR=[]
            TNR=[]
            PPV=[]
            F1=[]
            support=[]
            sum_arr = np.zeros((len(classes),len(classes)))
            for patient in patients[evaluation_method]:
                patient = str(patient)
                try:
                    if patient=='':
                        cm_path = os.path.join(results_path, segmentation_method, evaluation_method+"patient" , "CM_"+model+".pkl")
                    else:
                        cm_path = os.path.join(results_path, segmentation_method, evaluation_method+"patient" , patient, "CM_"+model+".pkl")
                    print(cm_path)
                    with open(cm_path, "rb") as f:
                        #print(os.path.join(results_path, model, "results-"+evaluation_method+"patient"+patient, "CM_"+segmentation_method+".pkl"))
                        arr = pickle.load(f)
                        print(arr)
                        if arr[0] is None:
                            continue
                except OSError as e:
                    skip=True
                    print(cm_path+" not found")
                    continue

                sum_arr+=arr    
            print(sum_arr)               
            acc, sensitivity, specificity, ppv, f1, sup = evaluate_metrics(sum_arr)
            # ACC.append(acc)
            # TPR.append(sensitivity)
            # TNR.append(specificity)
            # PPV.append(ppv)
            # F1.append(f1)
            # support.append(sup)
            if skip:
                continue
            metric_results = {"ACC":acc, "TPR": sensitivity, "PPV":ppv, "TNR":specificity, "F1":f1, "support":sup}
            #print(ACC)
            for metric in metrics:
            #print(name)
                #arr=np.array(metric_results[metric])
                df = pd.DataFrame(arr, columns=classes, index=np.arange(1,arr.shape[0]+1))
                #df1 = df.copy()
                df.loc["mean"]=metric_results[metric]#df1.mean()
                #df.loc["std"]=df1.std()  
                #df.loc["min"]=df1.min()
                #df.loc["max"]=df1.max() 
                #print(df)
                agg_metric = 0
                for i,cl in enumerate(classes):   
                    mean = df.loc['mean',cl]
                    #std = df.loc['std',cl]
                    #mmin = df.loc['min',cl]
                    #mmax = df.loc['max',cl]
                    #print(mean)
                    results[evaluation_method].loc[(model, segmentation_method),(cl, metric,"mean")] = mean
                    agg_metric += mean * metric_results["support"][i] / np.sum(metric_results["support"])
                    #results[evaluation_method].loc[(model, segmentation_method),(cl, metric,"std")] = std
                    #results[evaluation_method].loc[(model, segmentation_method),(cl, metric,"min")] = mmin
                    #results[evaluation_method].loc[(model, segmentation_method),(cl, metric,"max")] = mmax
                results[evaluation_method].loc[(model, segmentation_method),("All", metric,"mean")] = agg_metric
                #results[evaluation_method].loc[(model, segmentation_method),("All", metric,"std")] = np.mean(df.loc['std'], axis=0)
                #results[evaluation_method].loc[(model, segmentation_method),("All", metric,"min")] = np.mean(df.loc['min'], axis=0)
                #results[evaluation_method].loc[(model, segmentation_method),("All", metric,"max")] = np.mean(df.loc['max'], axis=0)
for evaluation_method in experiments:
    print(evaluation_method)
    results[evaluation_method] = results[evaluation_method].round(4)
    print(results[evaluation_method])
    results[evaluation_method].to_csv(os.path.join(results_path, "summary_"+evaluation_method+".csv"))
# remove macro file
# add intra- and inter- in folder name
