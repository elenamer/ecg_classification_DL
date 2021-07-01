
'''

Adapted from:

https://github.com/kweimann/ecg-transfer-learning/blob/master/transplant/modules/resnet1d.py


'''

import tensorflow as tf


def batch_norm():
    return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)


def relu():
    return tf.keras.layers.ReLU()

def dropout(dp=0):
    return tf.keras.layers.Dropout(dp)

def conv1d(filters, kernel_size=3, strides=1):
    return tf.keras.layers.Conv1D(
        filters, kernel_size, strides=strides, padding='same', use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling())


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, dropout = 0, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout = dropout

    def build(self, input_shape):
        num_chan = input_shape[-1]
        self.conv1 = conv1d(self.filters, self.kernel_size, self.strides)
        self.bn1 = batch_norm()
        self.relu1 = relu()
        self.dropout1 = dropout(self.dropout)
        self.conv2 = conv1d(self.filters, self.kernel_size, 1)
        self.bn2 = batch_norm()
        self.relu2 = relu()
        #self.dropout2 = dropout(self.dropout)
        if num_chan != self.filters or self.strides > 1:
            self.proj_conv = conv1d(self.filters, 1, self.strides)
            self.proj_bn = batch_norm()
            self.projection = True
        else:
            self.projection = False
        super().build(input_shape)

    def call(self, x, **kwargs):
        shortcut = x
        if self.projection:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_bn(shortcut)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x + shortcut)
        #x = self.dropout2(x + shortcut)
        return x


class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, expansion=4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expansion = expansion

    def build(self, input_shape):
        num_chan = input_shape[-1]
        self.conv1 = conv1d(self.filters, 1, 1)
        self.bn1 = batch_norm()
        self.relu1 = relu()
        self.conv2 = conv1d(self.filters, self.kernel_size, self.strides)
        self.bn2 = batch_norm()
        self.relu2 = relu()
        self.conv3 = conv1d(self.filters * self.expansion, 1, 1)
        self.bn3 = batch_norm()
        self.relu3 = relu()
        if num_chan != self.filters * self.expansion or self.strides > 1:
            self.proj_conv = conv1d(self.filters * self.expansion, 1, self.strides)
            self.proj_bn = batch_norm()
            self.projection = True
        else:
            self.projection = False
        super().build(input_shape)

    def call(self, x, **kwargs):
        shortcut = x
        if self.projection:
            shortcut = self.proj_conv(shortcut)
            shortcut = self.proj_bn(shortcut)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x + shortcut)
        return x


class ResNet(tf.keras.layers.Layer):
    def __init__(self, blocks=(2, 2, 2, 2),
                 filters=(64, 128, 256, 512), kernel_size=(3, 3, 3, 3),
                 block_fn=ResidualBlock, dropout = 0, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.filters = filters
        self.block_fn=block_fn
        self.block_nums = blocks
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.loss = 'categorical_crossentropy'

    def build(self, input_shape):
        self.conv1 = conv1d(64, 7, 2)
        self.bn1 = batch_norm()
        self.relu1 = relu()
        self.maxpool1 = tf.keras.layers.MaxPooling1D(3, 2, padding='same')
        self.blocks = []
        for stage, num_blocks in enumerate(self.block_nums):
            for block in range(num_blocks):
                strides = 2 if block == 0 and stage > 0 else 1
                res_block = self.block_fn(self.filters[stage], self.kernel_size[stage], strides, self.dropout)
                self.blocks.append(res_block)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()

        super().build(input_shape)

    def get_optimizer(self, lr):
        return tf.keras.optimizers.Adam(lr=lr)

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        for res_block in self.blocks:
            x = res_block(x)
        x = self.global_pool(x)
        return x
