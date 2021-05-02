from models.resnet import ResNet
from evaluation.metrics import F1Metric
from datasets.ptbxldataset import PTBXLDataset
from datasets.arr10000dataset import Arr10000Dataset
#from datasets.datagenerator import DataGenerator
from datasets.mitbihardataset import MITBIHARDataset
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import numpy as np

seed = 102

classes = ["N", "S", "V", "F", "Q"]

#classes = "rhythm"

num_classes = 5

db_name = "mitdb"
choice = "static"
eval_p = "intra"

dat = MITBIHARDataset(db_name)

## Warning: for now balance = True and False are treated the same and saved to same files (i.e. overwritten)
dat.generate_train_set(eval_p,choice,False)


dataset = dat.load_dataset(eval_p,choice, 'train', '' )

if len(dataset) != 2:
    print("generate crossval splits")
    # we need to do random n-crossval splits for val and test

skf = StratifiedKFold(n_splits=10)
for i, (train_val_ind, test_ind) in enumerate(skf.split(dataset[0],dataset[1])):
    # experiments/mitdb/static/specificpatient/201/models/"
    exp_path = "experiments"+os.sep+db_name+os.sep+choice+os.sep+eval_p+"patient"+os.sep+str(i)
    if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            os.mkdir(exp_path+os.sep+"models")
            os.mkdir(exp_path+os.sep+"results")

    X_test = dataset[0][test_ind]
    y_test = dataset[1][test_ind]
    train_val = (np.array(dataset[0][train_val_ind]), np.array(dataset[1][train_val_ind]))


    X_train, X_val, y_train, y_val = train_test_split(train_val[0], train_val[1], random_state=seed, shuffle=True, test_size=0.15)


    X_train, y_train = dat.balance(X_train, y_train)
    X_train, y_train = np.array(X_train), np.array(y_train)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(classes))

    model = ResNet(num_outputs=num_classes, blocks=[1,1], filters=[32, 64], kernel_size=[15,15], dropout=0.2)

    inputs = tf.keras.layers.Input((200,1,), dtype='float32')
    m1 = tf.keras.Model(inputs=inputs, outputs=model.call(inputs))

    opt = tf.keras.optimizers.Adam(lr=0.0001)

    m1.compile(optimizer=opt,
                #tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss='categorical_crossentropy',
                metrics='acc')

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    log_f1 = F1Metric(train=(X_train, y_train), validation=(X_val, y_val), path=exp_path+os.sep+"models")


    m1.fit(x=X_train,y=y_train, validation_data = (X_val, y_val), callbacks = [es, log_f1], epochs = 100)

    y_pred = m1.predict(X_test)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1),labels=range(num_classes))
    output = open(exp_path+os.sep+'CM_test.pkl', 'wb')
    pickle.dump(cm, output)
    output.close()

    y_pred = m1.predict(X_val)
    cm = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1),labels=range(num_classes))
    output = open(exp_path+os.sep+'CM_val.pkl', 'wb')
    pickle.dump(cm, output)
    output.close()
