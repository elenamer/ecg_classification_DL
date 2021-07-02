import tensorflow as tf
import os
from evaluation.metrics import F1Metric
from wandb.keras import WandbCallback
import numpy as np

class Classifier(tf.keras.Model):
    def __init__(self, model, input_seconds, frequency, n_classes=10, learning_rate=0.0001, epochs=20, path="temp"):
        super(Classifier, self).__init__()
        self.num_classes = n_classes
        self.learning_rate=learning_rate
        self.input_size = int(input_seconds * frequency)
        self.model = model#tf.keras.Model(inputs=inputs, outputs=model.call(inputs))
        out_act = 'sigmoid' if n_classes == 1 else 'softmax' # change this for multi-label
        self.classifier = tf.keras.layers.Dense(n_classes, out_act)
        os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
        self.epochs = epochs
        self.path = path
        

    def add_compile(self):
        self.compile(optimizer=self.model.get_optimizer(self.learning_rate),
        loss=self.model.loss,
        metrics='acc')

    def summary(self):
        input_layer = tf.keras.layers.Input(shape=(self.input_size,1,), dtype='float32')
        model = tf.keras.Model(inputs=input_layer, outputs=self.call(input_layer))
        return model.summary()


    def call(self, x, **kwargs):
        print(x.shape)
        x = self.model(x)
        x = self.classifier(x)
        return x
    
    def fit(self, x, y, validation_data):

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        log_f1 = F1Metric(train=(x, y), validation=validation_data, path=self.path+os.sep+"models")
        wandb_cb = WandbCallback(save_weights_only=True)

        x, y, idxs = process(X=x,labels=y,input_size=self.input_size, window=True)

        X_val, y_val, idxs = process(X = validation_data[0], labels=validation_data[1],input_size=self.input_size, window=True)
        super(Classifier, self).fit(x, y, validation_data = (X_val, y_val), callbacks = [es, log_f1, wandb_cb], epochs = self.epochs)

    def predict(self, X):
        X, idxs = process(X=X,input_size=self.input_size, window=True)

        preds = super().predict(X) # always on window-level
        agg_preds = aggregate_labels(preds, idxs)
        return agg_preds # always on set level



# These should be in a transform class


def aggregate_labels(preds, idmap):
    aggregate_fn = np.mean
    if(idmap is not None and len(idmap)!=len(np.unique(idmap))):
        print("aggregating predictions...")
        preds_aggregated = []
        targs_aggregated = []
        for i in np.unique(idmap):
            preds_local = preds[np.where(idmap==i)[0]]
            preds_aggregated.append(aggregate_fn(preds_local,axis=0))
        return np.array(preds_aggregated)

def pad(sig, size):
    return np.pad(np.array(sig), (0, size - len(sig)))

def process(X, input_size, labels=None, window=False):
    overlap = 0.5
    print("windowing")
    if window==True:
        new_data = []
        new_labels = []
        idmap = []
        for ind, sig in enumerate(X):
            #print(sig)
            step = int(input_size*overlap)
            nrows = ((len(sig)-input_size)//step)+1
            print(nrows)
            windows = sig[step*np.arange(nrows)[:,None] + np.arange(input_size)]
            #print(windows)
            new_data.extend(windows.tolist())
            if labels is not None:
                new_labels.extend([labels[ind]] * nrows)
            idmap.extend([ind] * nrows)
            #print(idmap)
    new_data = tf.keras.preprocessing.sequence.pad_sequences(new_data, maxlen=input_size, dtype="float32", padding="post", truncating="post", value=0.0)
    if new_data.ndim == 2:
        new_data = new_data[:,:,None]
    print(new_data.shape)
    if labels is None:
        return new_data, idmap
        # just crop/pad if needed
        # convert to numpy array

    new_labels = np.array(new_labels)
    print(new_labels.shape)
    return new_data, new_labels, idmap
