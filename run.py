
from datasets.cpsc2018dataset import CPSC2018Dataset
from processing.transform import Transform
from models.cpsc2018winner import CPSCWinnerNet
from models.transfer2020resnet import ResNet
from processing.slidingwindow import SlidingWindow
from datasets.ptbxldataset import PTBXLDataset
from evaluation.experiment import Experiment 


exp_config = {}

exp2 = Experiment(CPSC2018Dataset, Transform, 144, CPSCWinnerNet, 'rhythm', 'inter', 100) # learning parameters not passed for now?
                                                                                # do they depend on dataset? on model?

exp2.run()
exp2.evaluate()

exp1 = Experiment(PTBXLDataset, SlidingWindow, 2.5, ResNet, 'rhythm', 'inter', 100) # learning parameters not passed for now?
                                                                                # do they depend on dataset? on model?

exp1.run()
exp1.evaluate()


