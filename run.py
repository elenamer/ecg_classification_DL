
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


# exp_config = {}

# exp1 = Experiment(ResNet, SegmentBeats, 0.55, CNN, 'beat', 'inter', 100) # learning parameters not passed for now?
#                                                                                 # do they depend on dataset? on model?
# exp1.run()
# exp1.evaluate()

# exp2 = Experiment(CPSC2018Dataset, Transform, 144, CPSCWinnerNet, 'beat', 'inter', 100) # learning parameters not passed for now?
#                                                                                 # do they depend on dataset? on model?

# exp2.run()
# exp2.evaluate()

combinations_dict = {
    "form" : [(CPSC2018Dataset, 30, 60), (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    "rhythm" : [ (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    'cinc2017' : [(CPSC2018Dataset, 30, 60), (CincChallenge2017Dataset, 10, 30), (PTBXLDataset, 10, 10), (Arr10000Dataset, 10, 10)],
    "cpsc2018" : [(CPSC2018Dataset, 30, 60), (PTBXLDataset, 10, 10)]   
}

for model, sec in [(CNN, 10), (RTACNN,30), (CPSCWinnerNet, 144), (ResNet, 2.5), (WaveletModel, 10) ]:
    for task in combinations_dict.keys():
        for dat, threshold, alternative_sec in combinations_dict[task]:
            if sec >= threshold:
                exp3 = Experiment(dat, Transform, threshold, model, task, 'inter', 100)
                exp3.run()
                exp3.evaluate()

                exp3 = Experiment(dat, Transform, sec, model, task, 'inter', 100)
                exp3.run()
                exp3.evaluate()
            else:
                exp3 = Experiment(dat, Transform, alternative_sec, model, task, 'inter', 100)
                exp3.run()
                exp3.evaluate()
                