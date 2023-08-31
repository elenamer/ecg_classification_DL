# ECGDL: A framework for comparative study of databases and computational methods for arrhythmia detection from single-lead ECG

More details about this framework can be found in our [publication](https://www.nature.com/articles/s41598-023-38532-9).

This repository puts multiple ECG datasets for arrhythmia detection and arrhythmia detection methods under one umbrella. The focus is on using only one ECG lead in all methods. Initially, two different arrhythmia tasks are considered: *rhythm* classification and *heartbeat form* classification. 

## Usage

#### Environment

The requirements.txt file provides all the dependencies to run this framework. 

#### Datasets

Using [this script](https://github.com/elenamer/ecg_classification_DL/blob/8c1463daef7c21a0413f6e3830f8d60b7299cdc8/datasets/download_datasets.sh), all the datasets covered in this framework will be downloaded within a 'data' folder.

#### Tasks
Custom classification tasks can be defined as subsets of target arrhythmia classes. For this, please refer to the [label dictionary](https://github.com/elenamer/ecg_classification_DL/blob/f51e733954779774cd116bb41f6b5cf6a17144d5/tasks/class_mappings.csv), where a mapping of all relevant arrhythmia labels in each dataset is provided. They are mapped to common arrhythmia types. 

Custom tasks can be defined as .csv files in the /tasks folder. New tasks should be named in the following format: *newname*-task.csv (an example is the [form task file](https://github.com/elenamer/ecg_classification_DL/blob/f51e733954779774cd116bb41f6b5cf6a17144d5/tasks/form-task.csv)).

#### Experiments

The main class in this framework is the Experiment class. An example of defining Experiments can be found in [this script](https://github.com/elenamer/ecg_classification_DL/blob/f51e733954779774cd116bb41f6b5cf6a17144d5/run.py) for recording-level datasets and in [this script](https://github.com/elenamer/ecg_classification_DL/blob/f51e733954779774cd116bb41f6b5cf6a17144d5/run_physionet.py) for beat level datasets. More information about the categorization of the datasets and the implemented methodology can be found in our publication. 

## Experimental results

Results from experiments for arrhythmia classification can be found in our publication, as well as [here](https://github.com/elenamer/ecg_classification_DL/blob/146d51dcebae66c873d428406cc397477b0d4acc/results_tables/README.md).


## Citation

If this framework was useful in your work, please consider citing our paper:

```
@article{merdjanovska2023framework,
  title={A framework for comparative study of databases and computational methods for arrhythmia detection from single-lead ECG},
  author={Merdjanovska, Elena and Rashkovska, Aleksandra},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={11682},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
