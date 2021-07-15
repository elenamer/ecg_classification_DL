
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

# exp1 = Experiment(MITBIHARDataset, SegmentBeats, 0.72, CNN, 'beat', 'inter', 100) # learning parameters not passed for now?
#                                                                                 # do they depend on dataset? on model?
# exp1.run()
# exp1.evaluate()

# exp2 = Experiment(CPSC2018Dataset, Transform, 144, CPSCWinnerNet, 'beat', 'inter', 100) # learning parameters not passed for now?
#                                                                                 # do they depend on dataset? on model?

# exp2.run()
# exp2.evaluate()

exp3 = Experiment(CincChallenge2017Dataset, Transform, 30, RTACNN, 'rhythm', 'inter', 100) # learning parameters not passed for now?
                                                                                # do they depend on dataset? on model?

exp3.run()
exp3.evaluate()



