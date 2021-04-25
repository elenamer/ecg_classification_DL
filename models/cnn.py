import tensorflow as tf

class CNN(tf.keras.Model):

    def __init__(self, num_classes=3):
        super(CNN, self).__init__()

		# define layers
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):

		# call layers, forward pass
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)

