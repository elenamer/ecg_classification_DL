
from datasets.longtermafdataset import LongTermAFDataset
from datasets.mitbihsvdataset import MITBIHSVDataset
from datasets.incartdataset import INCARTDataset
from processing.repeatcrop import RepeatCrop
from datasets.arr10000dataset import Arr10000Dataset
from datasets.mitbihafdataset import AFDataset
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

combinations_dict_episodes = {
    "aami": [(MITBIHARDataset), (INCARTDataset), (MITBIHSVDataset), (LongTermAFDataset)]
    #"rhythm" : [(MITBIHARDataset, 10, 10)]#, (Arr10000Dataset, 10, 10)],
    #'cinc2017' : [(CPSC2018Dataset, 30, 60), (CincChallenge2017Dataset, 10, 30), (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    #"cpsc2018" : [(CPSC2018Dataset, 30, 60), (PTBXLDataset, 10, 10)]   
}


combinations_dict_beats = {
    "aami": [(LongTermAFDataset)]
    #"rhythm" : [(MITBIHARDataset, 10, 10)]#, (Arr10000Dataset, 10, 10)],
    #'cinc2017' : [(CPSC2018Dataset, 30, 60), (CincChallenge2017Dataset, 10, 30), (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    #"cpsc2018" : [(CPSC2018Dataset, 30, 60), (PTBXLDataset, 10, 10)]   
}

# if physionet dataset & rhythm task => segment episodes ( Transform/SlidingWindow and flag=true)
# if physionet dataset & form task => segment beats (SegmentBeats and flag=false)

for model, sec, freq in [(CNN, 0.72, 360),(CPSCWinnerNet, 0.72, 500), (ResNet, 0.72, 250),(RTACNN, 0.72, 300)]: #(CNN, 10, 360), (RTACNN, 30, 300), (CPSCWinnerNet, 144, 500), (ResNet, 2.5, 250), (WaveletModel, 10, 100) ]:
    for task in combinations_dict_beats.keys():
        for dat in combinations_dict_beats[task]:
            exp3 = Experiment(dat, Transform, freq, sec, model, task, 'intra', 100, episodes=False)
            exp3.run()
            exp3.evaluate()

            exp3 = Experiment(dat, Transform, freq, sec, model, task, 'inter', 100, episodes=False)
            exp3.run()
            exp3.evaluate()


exp3 = Experiment(dat, Transform, 500,  0.72, CPSCWinnerNet, 'aami', 'intra', 100, episodes=False)
exp3.run()
exp3.evaluate()


for model, sec, freq in [(CNN, 10, 360),(CPSCWinnerNet, 10, 500), (ResNet, 10, 250),(RTACNN, 10, 300)]: #(CNN, 10, 360), (RTACNN, 30, 300), (CPSCWinnerNet, 144, 500), (ResNet, 2.5, 250), (WaveletModel, 10, 100) ]:
    for task in combinations_dict_episodes.keys():
        for dat in combinations_dict_episodes[task]:
            exp3 = Experiment(dat, Transform, freq, sec, model, task, 'intra', 100, episodes=True, max_episode_seconds=sec)
            exp3.run()
            exp3.evaluate()
            
            exp3 = Experiment(dat, Transform, freq, sec, model, task, 'inter', 100, episodes=True, max_episode_seconds=sec)
            exp3.run()
            exp3.evaluate()

            if freq==300:
                exp3 = Experiment(dat, SlidingWindow, freq, sec, model, task, 'intra', 100, episodes=True, max_episode_seconds=2.5)
                exp3.run()
                exp3.evaluate()
                
                exp3 = Experiment(dat, SlidingWindow, freq, sec, model, task, 'inter', 100, episodes=True, max_episode_seconds=2.5)
                exp3.run()
                exp3.evaluate()



for model, sec, freq in [(CNN, 2.5, 360),(CPSCWinnerNet, 2.5, 500), (ResNet, 2.5, 250),(RTACNN, 2.5, 300)]: #(CNN, 10, 360), (RTACNN, 30, 300), (CPSCWinnerNet, 144, 500), (ResNet, 2.5, 250), (WaveletModel, 10, 100) ]:
    for task in combinations_dict_episodes.keys():
        for dat in combinations_dict_episodes[task]:
            exp3 = Experiment(dat, Transform, freq, sec, model, task, 'intra', 100, episodes=True, max_episode_seconds=sec)
            exp3.run()
            exp3.evaluate()

            exp3 = Experiment(dat, Transform, freq, sec, model, task, 'inter', 100, episodes=True, max_episode_seconds=sec)
            exp3.run()
            exp3.evaluate()