#Something like transforms, have it as an argument in dataset and call it

from types import CoroutineType
import tensorflow as tf
import os
from evaluation.metrics import F1Metric
from wandb.keras import WandbCallback


class Transform():


    def aggregate_labels(self, preds, idmap):
        raise NotImplementedError


    def process(self, X, input_size, labels=None, window=False):
        # keep padding/trunc and shape transform here
        # have separate subclass SlidingWindow for the other things
        raise NotImplementedError


    # def call(input signals/patient ids):
    #     maybe cropping
    #     maybe padding
    #     maaaaybe do label encoding here?
    #     return segmented beats/windows