
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

for model, sec in [(CNN, 10), (RTACNN,30), (CPSCWinnerNet, 144), (ResNet, 2.5), (WaveletModel, 10) ]:
    for task in ["rhythm", "form"]:
        if sec > 30:
            exp3 = Experiment(CPSC2018Dataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp3.run()
            exp3.evaluate()

            exp4 = Experiment(CPSC2018Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp4.run()
            exp4.evaluate()

        else:
            exp3 = Experiment(CPSC2018Dataset, Transform, 60, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp3.run()
            exp3.evaluate()

            exp4 = Experiment(CPSC2018Dataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp4.run()
            exp4.evaluate()      

            exp4 = Experiment(CPSC2018Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp4.run()
            exp4.evaluate()      

        if sec > 10: 

            exp3 = Experiment(PTBXLDataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp3.run()
            exp3.evaluate()

            exp3 = Experiment(PTBXLDataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp3.run()
            exp3.evaluate()

            exp4 = Experiment(PTBXLDataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp4.run()
            exp4.evaluate()
            
            exp3 = Experiment(Arr10000Dataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp3.run()
            exp3.evaluate()

            exp3 = Experiment(Arr10000Dataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp3.run()
            exp3.evaluate()

            exp4 = Experiment(Arr10000Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp4.run()
            exp4.evaluate()
        else:

            exp3 = Experiment(PTBXLDataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp3.run()
            exp3.evaluate()

            exp4 = Experiment(PTBXLDataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp4.run()
            exp4.evaluate()
            
            exp3 = Experiment(Arr10000Dataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp3.run()
            exp3.evaluate()

            exp4 = Experiment(Arr10000Dataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                            # do they depend on dataset? on model?
            exp4.run()
            exp4.evaluate()



for model, sec in [(CNN, 10), (RTACNN,30), (CPSCWinnerNet, 144), (ResNet, 2.5), (WaveletModel, 10) ]:
    task = 'cinc2017'  
    if sec > 30:
        exp3 = Experiment(CPSC2018Dataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(CPSC2018Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()

    else:
        exp3 = Experiment(CPSC2018Dataset, Transform, 60, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(CPSC2018Dataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()      

        exp4 = Experiment(CPSC2018Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()      

    if sec > 10: 

        exp3 = Experiment(PTBXLDataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp3 = Experiment(PTBXLDataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(PTBXLDataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()
        
        exp3 = Experiment(Arr10000Dataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp3 = Experiment(Arr10000Dataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(Arr10000Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()
    else:

        exp3 = Experiment(PTBXLDataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(PTBXLDataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()
        
        exp3 = Experiment(Arr10000Dataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(Arr10000Dataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()
         
    if sec > 10:    
        exp3 = Experiment(CincChallenge2017Dataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(CincChallenge2017Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()
    else:
        exp3 = Experiment(CincChallenge2017Dataset, Transform, 30, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(CincChallenge2017Dataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()


for model, sec in [(CNN, 10), (RTACNN,30), (CPSCWinnerNet, 144), (ResNet, 2.5), (WaveletModel, 10) ]:
    task='cpsc2018'
    if sec > 30:
        exp3 = Experiment(CPSC2018Dataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(CPSC2018Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()

    else:
        exp3 = Experiment(CPSC2018Dataset, Transform, 60, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(CPSC2018Dataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()      

        exp4 = Experiment(CPSC2018Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()      

    if sec > 10: 

        exp3 = Experiment(PTBXLDataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp3 = Experiment(PTBXLDataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(PTBXLDataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()
        
        exp3 = Experiment(Arr10000Dataset, Transform, sec, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp3 = Experiment(Arr10000Dataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(Arr10000Dataset, SlidingWindow, 2.5, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()
    else:

        exp3 = Experiment(PTBXLDataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(PTBXLDataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()
        
        exp3 = Experiment(Arr10000Dataset, Transform, 10, model, task, 'inter', 100) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp3.run()
        exp3.evaluate()

        exp4 = Experiment(Arr10000Dataset, SlidingWindow, sec, model, task, 'inter', 100, aggregate=True) # learning parameters not passed for now?
                                                                                        # do they depend on dataset? on model?
        exp4.run()
        exp4.evaluate()