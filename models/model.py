import tensorflow as tf
import os
from evaluation.metrics import F1Metric
from wandb.keras import WandbCallback
import numpy as np

class Classifier(tf.keras.Model):
    def __init__(self, model, input_seconds, frequency, n_classes, transform, learning_rate=0.0001, epochs=20, path="temp"):
        super(Classifier, self).__init__()
        self.num_classes = n_classes
        self.learning_rate=learning_rate
        self.input_size = int(input_seconds * frequency)
        self.model = model#tf.keras.Model(inputs=inputs, outputs=model.call(inputs))
        self.transform = transform(self.input_size) ## probably needs to move this outside of model(classifier), or pass parameters as *args
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
        #log_f1 = F1Metric(train=(x, y), validation=(X_val, y_val), path=self.path+os.sep+"models")
        wandb_cb = WandbCallback(save_weights_only=True)

        x, y = self.transform.process(X=x,labels=y,window=True)

        X_val, y_val = self.transform.process(X = validation_data[0], labels=validation_data[1],window=True)
        super(Classifier, self).fit(x, y, validation_data = (X_val, y_val), callbacks = [es, wandb_cb], epochs = self.epochs)

    def predict(self, X, y):
        # y is only R-peak locations in this case
        self.transform.reset_idmap()
        X = self.transform.process(X=X, labels = y, window=True)

        preds = super(Classifier, self).predict(X) # always on window-level
        agg_preds = self.transform.aggregate_labels(preds)
        return agg_preds # always on set level

