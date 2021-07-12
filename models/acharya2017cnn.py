
'''
Adapted from: https://github.com/on-device-ai/ClassifyHeartbeats/blob/master/tensorflow/classify_heartbeats_cnn.ipynb

'''

import tensorflow as tf

class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = num_filters
        self.kernel = kernel_size

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv1D(self.filters, self.kernel, activation='linear')
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling1D(2,2)
        super().build(input_shape)

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x




class CNN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.num_blocks = 3
        self.kernel_sizes = [3,4,4]
        self.num_filters = [5,10,20]
        self.loss = 'categorical_crossentropy' #?
        self.model_name = "cnn"

    def build(self, input_shape):
        # self.reshape = tf.keras.layers.Reshape((input_shape.shape[1], 1))
        self.blocks = []
        for id, filters in enumerate(self.num_filters):
            block = CNNBlock(filters, self.kernel_sizes[id])
            self.blocks.append(block)
        self.flat = tf.keras.layers.Flatten()
        self.dense1  = tf.keras.layers.Dense(30, activation='linear')
        self.relu1  = tf.keras.layers.ReLU()
        self.dense2  = tf.keras.layers.Dense(20, activation='linear')
        self.relu2  = tf.keras.layers.ReLU()
        super().build(input_shape)

    def get_optimizer(self, lr):
        return tf.keras.optimizers.Adam(lr=lr)

    def call(self, x, **kwargs):
        # x = self.reshape(x)
        for block in self.blocks:
            x = block(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        return x

    @staticmethod
    def get_name():
        print("INSIDE GET NAME")
        return 'cnn'

