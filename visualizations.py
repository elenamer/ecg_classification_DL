from datasets.savvydataset import SavvyDataset
from datasets.longtermafdataset import LongTermAFDataset
from datasets.mitbihsvdataset import MITBIHSVDataset
from datasets.mitbihardataset import MITBIHARDataset
from datasets.physionetdataset import PhysionetDataset
from datasets.ptbxldataset import PTBXLDataset
from datasets.cinc2017dataset import CincChallenge2017Dataset
from datasets.irhythmtestdataset import iRhythmTestDataset
from datasets.cpsc2018dataset import CPSC2018Dataset
from datasets.arr10000dataset import Arr10000Dataset
from datasets.incartdataset import INCARTDataset
from datasets.mitbihafdataset import AFDataset

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

dataset_names_dict = {
    'arr10000' : "Arr10000"


}


def classes_heatmap(path, task):
    dat = Arr10000Dataset(task=task)
    df = pd.DataFrame(columns = dat.class_names)
    cls = dat.data_distribution_tables()
    df = pd.concat([df, cls], axis=0)


    dat=PTBXLDataset(task=task)
    cls = dat.data_distribution_tables()
    df = pd.concat([df, cls], axis=0)

    dat= INCARTDataset(task=task)
    cls = dat.data_distribution_tables()
    df = pd.concat([df, cls], axis=0)

    dat= MITBIHARDataset(task=task)
    cls = dat.data_distribution_tables()
    df = pd.concat([df, cls], axis=0)

    dat= LongTermAFDataset(task=task)
    cls = dat.data_distribution_tables()
    df = pd.concat([df, cls], axis=0)

    dat= AFDataset(task=task)
    cls = dat.data_distribution_tables()
    df = pd.concat([df, cls], axis=0)

    dat= MITBIHSVDataset(task=task)
    cls = dat.data_distribution_tables()
    df = pd.concat([df, cls], axis=0)

    dat = CPSC2018Dataset(lead='V1-V2',task=task)
    cls = dat.data_distribution_tables()
    df = pd.concat([df, cls], axis=0)

    dat = CincChallenge2017Dataset(task=task)
    cls = dat.data_distribution_tables()
    df = pd.concat([df, cls], axis=0)

    print(df)


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
    df = df.astype(int)
    eps = 10e-1
    
    sns.heatmap(df+eps, annot=df.values, norm=LogNorm(), fmt="d")
    labels = axes.get_yticklabels()
    axes.set_yticklabels(labels, rotation=0) 
    fig.tight_layout()
    fig.savefig(path+os.sep+task+'-labels-heatmap.png', dpi=400)

# classes_heatmap("data_overviews", "all")
# classes_heatmap("data_overviews", "cinc2017")
classes_heatmap("data_overviews", "cpsc2018")