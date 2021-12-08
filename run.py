from processing.repeatcrop import RepeatCrop
from datasets.arr10000dataset import Arr10000Dataset
from datasets.cinc2017dataset import CincChallenge2017Dataset
from models.rtacnn import RTACNN
from models.ptb2020wavelet import WaveletModel
from models.acharya2017cnn import CNN
from processing.segmentbeats import SegmentBeats
from datasets.mitbihardataset import MITBIHARDataset
from datasets.cpsc2018dataset import CPSC2018Dataset
from processing.transform import Transform
from models.cpsc2018winner import CPSCWinnerNet
from models.transfer2020resnet import ResNet
from processing.slidingwindow import SlidingWindow
from datasets.ptbxldataset import PTBXLDataset
from evaluation.experiment import Experiment 
import os

os.environ["CUDA_VISIBLE_DEVICES"] ='1'


combinations_dict = {
    "form" : [(CPSC2018Dataset, 30, 60), (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    "rhythm" : [(PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    'cinc2017' : [(CPSC2018Dataset, 30, 60), (CincChallenge2017Dataset, 10, 30), (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    "cpsc2018" : [(CPSC2018Dataset, 30, 60), (PTBXLDataset, 10, 10)]   
}

combinations_dict1 = {
    #"form" : [(CPSC2018Dataset, 30, 60), (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    #"rhythm" : [(PTBXLDataset, 10, 10)]#, (Arr10000Dataset, 10, 10)],
    'aami' : [(CPSC2018Dataset, 30, 30), (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    #"cpsc2018" : [(CPSC2018Dataset, 30, 60), (PTBXLDataset, 10, 10)]   
}
 
for model, sec, freq in [(RTACNN, 30, 300),(ResNet, 2.5, 250),(CPSCWinnerNet, 30, 100),(CNN, 10, 360)]: #(CNN, 10, 360), (RTACNN, 30, 300), (CPSCWinnerNet, 144, 500), (ResNet, 2.5, 250), (Wav>
    for task in combinations_dict1.keys():
        for dat, threshold, alternative_sec in combinations_dict1[task]:
            if sec >= threshold:
                exp3 = Experiment(dat, Transform, freq,  threshold, model, task, 'inter', 100)
                exp3.run()
                exp3.evaluate()
            else:
                exp3 = Experiment(dat, SlidingWindow, freq, sec, model, task, 'inter', 100)
                exp3.run()
                exp3.evaluate()

