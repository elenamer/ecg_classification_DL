
from models.resnet import ResNet
from evaluation.metrics import F1Metric
from datasets.ptbxldataset import PTBXLDataset
from datasets.cinc2017dataset import CincChallenge2017Dataset
from datasets.cpsc2018dataset import CPSC2018Dataset
from datasets.arr10000dataset import Arr10000Dataset
import os
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import numpy as np


# dat = Arr10000Dataset()
# dat.data_distribution_tables()
# dat.get_signal(dat.path+os.sep+"ECGData", "MUSE_20180116_121716_47000.csv")
# print(dat.get_annotation("MUSE_20180116_121716_47000"))

seed = 102

#classes = "rhythm"

num_classes = 10

db_name = "ptb-xl"
choice = "static"
eval_p = "crossval"

dat=PTBXLDataset()

## Warning: for now balance = True and False are treated the same and saved to same files (i.e. overwritten)
X_train, Y_train, X_val, Y_val, X_test, Y_test = dat.get_crossval_splits(task="rhythm",split=9)

# experiments/mitdb/static/specificpatient/201/models/"
exp_path = "experiments"+os.sep+db_name+os.sep+choice+os.sep+eval_p+"patient"#+os.sep+str(i)
if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        os.mkdir(exp_path+os.sep+"models")
        os.mkdir(exp_path+os.sep+"results")

model = ResNet(num_outputs=num_classes, blocks=[1,1], filters=[32, 64], kernel_size=[15,15], dropout=0.1)

inputs = tf.keras.layers.Input((1000,1,), dtype='float32')
m1 = tf.keras.Model(inputs=inputs, outputs=model.call(inputs))
opt = tf.keras.optimizers.Adam(lr=0.0001)

m1.compile(optimizer=opt,
            #tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            loss='categorical_crossentropy',
            metrics='acc')

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
log_f1 = F1Metric(train=(X_train, Y_train), validation=(X_val, Y_val), path=exp_path+os.sep+"models")


m1.fit(x=X_train,y=Y_train, validation_data = (X_val, Y_val), callbacks = [es, log_f1], epochs = 20)

y_pred = m1.predict(X_test)
cm = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1),labels=range(num_classes))
output = open(exp_path+os.sep+'CM_test.pkl', 'wb')
pickle.dump(cm, output)
output.close()

# y_pred = m1.predict(X_val)
# cm = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1),labels=range(num_classes))
# output = open(exp_path+os.sep+'CM_val.pkl', 'wb')
# pickle.dump(cm, output)
# output.close()
