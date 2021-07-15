
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score

### Defining the Callback Metrics
class MetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, train, validation, path):   
        super(MetricCallback, self).__init__()
        self.validation = validation
        self.train = train
        self.root = path
        print('train shape', len(self.train[0]))
        print('validation shape', len(self.validation[0]))
        self.train_scores = []
        self.val_scores = []
        self.train_losses = []
        self.val_losses = []
     
    def on_epoch_end(self, epoch, logs={}):
        train_score, val_score = self.train_val_scores() 
        self.val_scores.append(val_score)
        self.train_scores.append(train_score)
        self.val_losses.append(logs.get('val_loss'))
        self.train_losses.append(logs.get('loss'))
        print(f'— train_{self.name}: {train_score} — val_{self.name}: {val_score} ')
    
    def on_train_end(self,logs={}):
        fig1, ax1 = plt.subplots()
        ax1.plot(self.train_scores)
        ax1.plot(self.val_scores)
        ax1.set_title('model '+self.name)
        ax1.set_ylabel(self.name)
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'validation'], loc='upper left')
        fig1.savefig(self.root+os.sep+self.name+'.png', dpi=450)
        plt.close(fig1)
        # summarize history for loss
        fig2, ax2 = plt.subplots()
        ax2.plot(self.train_losses)
        ax2.plot(self.val_losses)
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'validation'], loc='upper left')
        fig2.savefig(self.root+os.sep+'loss.png', dpi=450)
        plt.close(fig2)


class F1Metric(MetricCallback):
    def __init__(self, train, validation, path):   
        super(F1Metric, self).__init__(train, validation, path)
        self.name = "f1score"
     
    def train_val_scores(self):
        val_targ = self.validation[1].argmax(axis=1)
        print(self.validation[1].shape)
        print(self.validation[0].shape)
        val_predict = (np.asarray(self.model.predict(self.validation[0]).argmax(axis=1)))
        train_targ = self.train[1].argmax(axis=1)  
        # print(val_targ.shape)
        # print(val_predict.shape)
        train_predict = (np.asarray(self.model.predict(self.train[0]).argmax(axis=1)))
        val_f1 = round(f1_score(val_targ, val_predict, average='macro'), 6)
        train_f1 = round(f1_score(train_targ, train_predict, average='macro'), 6)
        return train_f1, val_f1


### Defining the Callback Metrics
class AUCMetric(MetricCallback):
    def __init__(self, train, validation, path):   
        super(AUCMetric, self).__init__(train, validation, path)
        self.name = "aucscore"
     
    def train_val_scores(self):
        val_targ = self.validation[1]
        # print(self.validation[1].shape)
        # print(self.validation[0].shape)
        # print(val_targ)
        val_predict = np.asarray(self.model.predict(self.validation[0]))
        # print(val_predict)
        train_targ = self.train[1] 
        train_predict = (np.asarray(self.model.predict(self.train[0])))
        # print(train_targ.sum(axis=0))
        # print(train_targ.sum(axis=0).shape)
        # print(train_predict.sum(axis=0))
        # print(train_predict.sum(axis=0).shape)
        # print(val_targ.sum(axis=0))
        # print(val_targ.sum(axis=0).shape)
        # print(val_predict.sum(axis=0))
        # print(val_predict.sum(axis=0).shape)
        labels = np.argwhere(train_targ.sum(axis=0) > 0 )
        print(labels[:,0])
        train_auc = round(roc_auc_score(train_targ[:,(labels[:,0])], train_predict[:,(labels[:,0])], average='macro'), 6)
        labels = np.argwhere(val_targ.sum(axis=0) > 0 )
        val_auc = round(roc_auc_score(val_targ[:,(labels[:,0])], val_predict[:,(labels[:,0])], average='macro'), 6)
        return train_auc, val_auc

        