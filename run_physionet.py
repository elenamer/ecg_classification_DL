
from datasets.mitbihsvdataset import MITBIHSVDataset
from datasets.incartdataset import INCARTDataset
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

combinations_dict = {
    "form" : [(MITBIHARDataset)],
    "aami": [(MITBIHARDataset), (INCARTDataset), (MITBIHSVDataset)]
    #"rhythm" : [(MITBIHARDataset, 10, 10)]#, (Arr10000Dataset, 10, 10)],
    #'cinc2017' : [(CPSC2018Dataset, 30, 60), (CincChallenge2017Dataset, 10, 30), (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    #"cpsc2018" : [(CPSC2018Dataset, 30, 60), (PTBXLDataset, 10, 10)]   
}


for model, sec, freq in [(CPSCWinnerNet, 0.72, 500), (ResNet, 0.72, 250), (CNN, 0.72, 360),(RTACNN, 30, 300)]: #(CNN, 10, 360), (RTACNN, 30, 300), (CPSCWinnerNet, 144, 500), (ResNet, 2.5, 250), (WaveletModel, 10, 100) ]:
    for task in combinations_dict.keys():
        for dat in combinations_dict[task]:
            exp3 = Experiment(dat, Transform, freq, sec, model, task, 'intra', 100)
            exp3.run()
            exp3.evaluate()

