import tensorflow as tf
import os
from evaluation.metrics import F1Metric
from wandb.keras import WandbCallback


class Classifier(tf.keras.Model):
    def __init__(self, model, input_size, n_classes=10, learning_rate=0.0001, epochs=20, path="temp"):
        super(Classifier, self).__init__()
        self.num_classes = n_classes
        self.learning_rate=learning_rate
        self.input_size = input_size
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
        super(Classifier, self).fit(x,y, validation_data = validation_data, callbacks = [es, log_f1, wandb_cb], epochs = self.epochs)
