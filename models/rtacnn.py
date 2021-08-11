import tensorflow as tf
from tensorflow.keras import  backend as K
from tensorflow.keras import initializers, regularizers


__all__ = ['RTA_CNN', 'VGG12', 'RESNET50', 'MSCNN', 'SENET', 'WDCNN']


def en_loss(y_true, y_pred):
    
    epsilon = 1.e-7
    gamma = float(0.3)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    pos_pred = tf.math.pow(-tf.math.log(y_pred),gamma)
    nag_pred = tf.math.pow(-tf.math.log(1-y_pred),gamma)
    y_t = tf.math.multiply(y_true, pos_pred) + tf.math.multiply(1-y_true, nag_pred)
    en_loss = tf.math.reduce_mean(y_t)
    return en_loss


# RTA-CNN

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, nb_filter, kernel_size):
        super().__init__()
        self.nb_filter = nb_filter
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv1D(self.nb_filter, self.kernel_size, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        super().build(input_shape)

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x) 
        x = self.relu(x)       
        return x

class AttentionBranch(tf.keras.layers.Layer):
    def __init__(self, nb_filter, kernel_size):
        super().__init__()
        self.nb_filter = nb_filter
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = ConvBlock(self.nb_filter, self.kernel_size)
        self.maxpool = tf.keras.layers.MaxPooling1D(2)
        self.conv2 = ConvBlock(self.nb_filter, self.kernel_size)
        self.upsamp = tf.keras.layers.UpSampling1D(size = 2)
        self.conv3 = ConvBlock(self.nb_filter, self.kernel_size)
        self.zp = tf.keras.layers.ZeroPadding1D(1)
        self.crop = tf.keras.layers.Cropping1D((1,0))
        self.conv4 = ConvBlock(self.nb_filter, self.kernel_size)
        self.conv_layer = tf.keras.layers.Conv1D(self.nb_filter, 1, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        super().build(input_shape)
        
    def call(self, x):
        x1 = self.conv1(x)

        x = self.maxpool(x1)
        x = self.conv2(x)
        x = self.upsamp(x)

        x2 = self.conv3(x)
        
        if(K.int_shape(x1)!=K.int_shape(x2)):
            x2 = self.zp(x2)
            x2 = self.crop(x2)

        x = tf.keras.layers.add([x1, x2])    

        x = self.conv4(x)

        x = self.conv_layer(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        
        return x

class RTABlock(tf.keras.layers.Layer):

    def __init__(self, nb_filter, kernel_size):
        super().__init__()
        self.nb_filter = nb_filter
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = ConvBlock(self.nb_filter, self.kernel_size)
        self.conv2 = ConvBlock(self.nb_filter, self.kernel_size)

        self.attention = AttentionBranch(self.nb_filter, self.kernel_size)
        self.conv3 = ConvBlock(self.nb_filter, self.kernel_size)
        super().build(input_shape)

    def call(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        attention_map = self.attention(x1)
        
        x = tf.keras.layers.multiply([x2, attention_map])
        x = tf.keras.layers.add([x, x1])
        
        out = self.conv3(x)
        
        return out

class RTACNN(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.loss = en_loss
        self.model_name = "rtacnn"

    def get_optimizer(self, lr):
        return tf.keras.optimizers.Adam(lr=lr) ## 0.0001

    def build(self, input_shape):
        self.rta1 = RTABlock(16, 32)
        self.maxpool1 = tf.keras.layers.MaxPooling1D(4)
        self.rta2 = RTABlock(32, 16)
        self.maxpool2 = tf.keras.layers.MaxPooling1D(4)
        self.rta3 = RTABlock(64, 9)
        self.maxpool3 = tf.keras.layers.MaxPooling1D(2)
        self.rta4 = RTABlock(64, 9)
        self.maxpool4 = tf.keras.layers.MaxPooling1D(2)
        self.dp1 = tf.keras.layers.Dropout(0.6)
        self.rta5 = RTABlock(128, 3)
        self.maxpool5 = tf.keras.layers.MaxPooling1D(2)
        self.rta6 = RTABlock(128, 3)
        self.maxpool6 = tf.keras.layers.MaxPooling1D(2)
        self.dp2 = tf.keras.layers.Dropout(0.6)
        self.flat = tf.keras.layers.Flatten()
        self.dp3 = tf.keras.layers.Dropout(0.7)
        self.dense3 = tf.keras.layers.Dense(100,  activation='relu')
        self.dp4 = tf.keras.layers.Dropout(0.7)
        super().build(input_shape)
        
    def call(self, x):
        x = self.rta1(x) 
        x = self.maxpool1(x) 
        x = self.rta2(x) 
        x = self.maxpool2(x) 
        x = self.rta3(x) 
        x = self.maxpool3(x) 
        x = self.rta4(x) 
        x = self.maxpool4(x) 
        x = self.dp1(x) 
        x = self.rta5(x) 
        x = self.maxpool5(x) 
        x = self.rta6(x) 
        x = self.maxpool6(x) 
        x = self.dp2(x) 
        x = self.flat(x) 
        x = self.dp3(x) 
        x = self.dense3(x) 
        x = self.dp4(x) 
        x = self.dense4(x)
        return x

        
    @staticmethod
    def get_name():
        return 'rtacnn'



#1DCNN
def WDCNN():
    
    inputs = Input((9000, 1))
    
    x = Conv1D(16, 32, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(32, 16, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(64, 9, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(64, 9, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(100,  activation='relu')(x)
    x = Dense(3,  activation='softmax')(x)
    
    model = Model(inputs, x)
    
    return model


# VGG12
def VGG12():
    
    inputs = Input((9000, 1))
    
    x = Conv1D(64, 3, padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv1D(64, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)

    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)

    x = Conv1D(512, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(512, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(512, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(4)(x)

    
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(450,  activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(3,  activation='softmax')(x)
    
    model = Model(inputs, x)
    
    return model

# RESNET50
def identity_block(in_x, nb_filters):

    F1, F2, F3 = nb_filters
    
    x = in_x
    
    x = Conv1D(F1, 1, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv1D(F2, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(F3, 1,  padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    
    x = add([x, in_x])
    x = Activation("relu")(x)
    
    return x

def convolutional_block(in_x, nb_filters, stride):

    F1, F2, F3 = nb_filters
    
    x = in_x

    x = Conv1D(F1, 1, strides=stride,  padding='same', kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(F2, 3, strides=1, padding='same', kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(F3, 1, strides=1, padding='same', kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    x1 = Conv1D(F3, 1, strides=stride, padding='same', kernel_initializer="he_normal")(in_x)
    x1 = BatchNormalization()(x1)

    x = add([x, x1])
    x = Activation('relu')(x)
    
    return x

def RESNET50():
    
    inputs = Input((9000, 1))

    filter_num = 64
    
    x = Conv1D(filter_num, 7, strides=2, kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(3, stride=2)(x)

    x = convolutional_block(x, [filter_num, filter_num, filter_num * 4], 4)
    x = identity_block(x, [filter_num, filter_num, filter_num * 4])
    x = identity_block(x, [filter_num, filter_num, filter_num * 4])
    
    x = convolutional_block(x, [filter_num * 2, filter_num * 2, filter_num * 8], 4)
    x = identity_block(x, [filter_num * 2, filter_num * 2, filter_num * 8])
    x = identity_block(x, [filter_num * 2, filter_num * 2, filter_num * 8])
    x = identity_block(x, [filter_num * 2, filter_num * 2, filter_num * 8])
    
    x = convolutional_block(x, [filter_num * 4, filter_num * 4, filter_num * 16], 2)
    x = identity_block(x, [filter_num * 4, filter_num * 4, filter_num * 16])
    x = identity_block(x, [filter_num * 4, filter_num * 4, filter_num * 16])
    x = identity_block(x, [filter_num * 4, filter_num * 4, filter_num * 16])
    x = identity_block(x, [filter_num * 4, filter_num * 4, filter_num * 16])
    x = identity_block(x, [filter_num * 4, filter_num * 4, filter_num * 16])
    
    x = convolutional_block(x, [filter_num * 8, filter_num * 8, filter_num * 32], 2)
    x = identity_block(x, [filter_num * 8, filter_num * 8, filter_num * 32])
    x = identity_block(x, [filter_num * 8, filter_num * 8, filter_num * 32])
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(500, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(3, activation="softmax")(x)
    
    model = Model(inputs, x)

    return model

#MSCNN
def MSCNN():
    
    inputs = Input((9000, 1))
    
    x1 = Conv1D(64, 3, padding='same')(inputs)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(64, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=3)(x1)
    
    x1 = Conv1D(128, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(128, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=3)(x1)
    
    x1 = Conv1D(256, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(256, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(256, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=2)(x1)
    
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=2)(x1)
    
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(512, 3, padding='same')(x1)
    x1 = Activation('relu')(x1)
    
    x1 = MaxPooling1D(2,stride=2)(x1)
    
    
    
    x2 = Conv1D(64, 3, padding='same')(inputs)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(64, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=3)(x2)
    
    x2 = Conv1D(128, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(128, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=3)(x2)
    
    x2 = Conv1D(256, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(256, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(256, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=2)(x2)
    
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=2)(x2)
    
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(512, 3, padding='same')(x2)
    x2 = Activation('relu')(x2)
    
    x2 = MaxPooling1D(2,stride=2)(x2)
    
    x = concatenate([x1 , x2] , axis=-1)
    
    
    x = Flatten()(x)
    x = Dense(256,  activation='relu')(x)
    x = Dense(3,  activation='softmax')(x)
    
    model = Model(inputs, x)
    
    return model

#SENET
def se_block(in_x, nb_filter):
        squeeze = GlobalAveragePooling1D()(in_x)
        excitation = Dense(units=nb_filter // 2)(squeeze)
        excitation = Activation('relu')(excitation)
        excitation = Dense(units=nb_filter)(excitation)
        excitation = Activation('sigmoid')(excitation)
        excitation = Reshape((1,nb_filter))(excitation)
        scale = multiply([in_x,excitation])
        out = add([in_x, scale])
        
        return out


def SENET():
    
    inputs = Input((9000, 1))
    
    x = Conv1D(16, 32, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x, 16)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(32, 16, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x, 32)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(64, 9, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x, 64)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(64, 9, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x, 64)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x, 128)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = se_block(x, 128)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(300,  activation='relu')(x)
    x = Dense(3,  activation='softmax')(x)
    
    model = Model(inputs, x)
    
    return model


