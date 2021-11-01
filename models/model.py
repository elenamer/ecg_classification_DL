import tensorflow as tf
import os
from .metriccallbacks import AUCMetric, F1Metric, TimeHistory
from wandb.keras import WandbCallback
import numpy as np

class Classifier(tf.keras.Model):
    def __init__(self, model, input_size, n_classes, learning_rate=0.0001, epochs=20, path="temp"):
        super(Classifier, self).__init__()
        self.num_classes = n_classes
        self.learning_rate=learning_rate
        self.input_size = input_size
        self.model = model#tf.keras.Model(inputs=inputs, outputs=model.call(inputs))
        out_act = 'sigmoid' # if n_classes == 1 else 'softmax' # change this for multi-label
        self.classifier = tf.keras.layers.Dense(n_classes, out_act)
        os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
        self.epochs = epochs
        self.path = path
        

    def add_compile(self):
        self.compile(optimizer=self.model.get_optimizer(self.learning_rate),
        loss=self.model.loss,
        metrics=['acc', tf.keras.metrics.AUC(multi_label=True)])

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
        time_callback = TimeHistory()

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        wandb_cb = WandbCallback(save_weights_only=True)

        # x, y = self.transform.process(X=x,labels=y)

        X_val, y_val = validation_data[0], validation_data[1]
        #log_f1 = F1Metric(train=(x, y), validation=(X_val, y_val), path=self.path+os.sep+"models")
        #log_auc = AUCMetric(train=(x, y), validation=(X_val, y_val), path=self.path+os.sep+"models")

        super(Classifier, self).fit(x, y, validation_data = (X_val, y_val), callbacks = [es, time_callback, wandb_cb], epochs = self.epochs, batch_size=128)
        times = time_callback.times
        return times
    # def predict(self, X, y):
    #     # y is only R-peak locations in this case


    #     preds = super(Classifier, self).predict(X) # always on window-level
    #     print(preds.shape)
    #     agg_preds = self.transform.aggregate_labels(preds)
    #     print(agg_preds.shape)
    #     return agg_preds # always on set level

