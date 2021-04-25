from datasets.physionetdataset import PhysionetDataset
from datasets.ptbxldataset import PTBXLDataset
from datasets.cinc2017dataset import CincChallenge2017Dataset
from datasets.irhythmtestdataset import iRhythmTestDataset
from datasets.cpsc2018dataset import CPSC2018Dataset
from datasets.arr10000dataset import Arr10000Dataset

dbs=[("incartdb",12),("afdb",2),('mitdb',2),("svdb",2), ("ltafdb",2)]

#for db, nc in dbs:
#    PhysionetDataset(db, nc)
#Arr10000Dataset()
#CPSC2018Dataset()
dat=PTBXLDataset()
#CincChallenge2017Dataset()
#iRhythmTestDataset()
dat.examine_database()