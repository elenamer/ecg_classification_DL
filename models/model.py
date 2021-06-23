import tensorflow as tf

class Classifier(tf.keras.Model):
    def __init__(self, model, input_size, n_classes=10, learning_rate=0.0001):
        super(Classifier, self).__init__()
        self.num_classes = n_classes
        self.learning_rate=learning_rate

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_size,1,), dtype='float32')
        self.model = model#tf.keras.Model(inputs=inputs, outputs=model.call(inputs))
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        out_act = 'sigmoid' if n_classes == 1 else 'softmax' # change this for multi-label
        self.classifier = tf.keras.layers.Dense(n_classes, out_act)


    def add_compile(self):
        self.compile(optimizer=self.model.get_optimizer(self.learning_rate),
        loss=self.model.loss,
        metrics='acc')


    def call(self, x, **kwargs):
        print(x.shape)
        x = self.model.call(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x