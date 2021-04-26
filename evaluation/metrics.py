
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve

### Defining the Callback Metrics
class F1Metric(tf.keras.callbacks.Callback):
    def __init__(self, train, validation, path):   
        super(F1Metric, self).__init__()
        self.validation = validation
        self.train = train
        self.root = path
        print('train shape', len(self.train[0]))
        print('validation shape', len(self.validation[0]))
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_losses = []
        self.val_losses = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1].argmax(axis=1)
        val_predict = (np.asarray(self.model.predict(self.validation[0]).argmax(axis=1)))
        train_targ = self.train[1].argmax(axis=1)  
        print(val_targ.shape)
        print(val_predict.shape)
        train_predict = (np.asarray(self.model.predict(self.train[0]).argmax(axis=1)))
        val_f1 = round(f1_score(val_targ, val_predict, average='macro'), 6)
        train_f1 = round(f1_score(train_targ, train_predict, average='macro'), 6)
        self.val_f1_scores.append(val_f1)
        self.train_f1_scores.append(train_f1)
        self.val_losses.append(logs.get('val_loss'))
        self.train_losses.append(logs.get('loss'))

        print(f'— train_f1: {train_f1} — val_f1: {val_f1} ')
    
    def on_train_end(self,logs={}):
        fig1, ax1 = plt.subplots()
        ax1.plot(self.train_f1_scores)
        ax1.plot(self.val_f1_scores)
        ax1.set_title('model f1score')
        ax1.set_ylabel('f1score')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'validation'], loc='upper left')
        fig1.savefig(self.root+'f1score.png', dpi=450)
        plt.close(fig1)
        # summarize history for loss
        fig2, ax2 = plt.subplots()
        ax2.plot(self.train_losses)
        ax2.plot(self.val_losses)
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'validation'], loc='upper left')
        fig2.savefig(self.root+'loss.png', dpi=450)
        plt.close(fig2)

