import tensorflow as tf
import tensorflow.keras.backend as K

class DenseNet121:
    def __init__(self, input_shape = (128,128,3), dp_rate = 0.3, n_cls = 12, growth_rate = 32):
        self.num_layers = [6, 12, 24, 16]
        self.input_shape = input_shape
        self.dp_rate = dp_rate
        self.n_cls = n_cls
        self.k = growth_rate

    def conv_block(self, l, filter, kernel, stride):
        l = tf.keras.layers.BatchNormalization()(l)
        l = tf.keras.layers.ReLU()(l)
        l = tf.keras.layers.Conv2D(filter, kernel, stride, padding = 'same')(l)
        return l

    def dense_block(self, pre_layer, num_layers):
        for layer in range(num_layers):
            # feature extraction -> produce 4 feature maps * growth rate
            l = self.conv_block(pre_layer, filter = 4*self.k, kernel = (1,1), stride = (1,1))
            # bringing down the depth -> 3*3 kernel
            l = self.conv_block(l, filter = self.k, kernel = (3,3), stride = (1,1))
            # concatenate layer
            pre_layer = tf.keras.layers.Concatenate()([pre_layer, l])
        return pre_layer


    def transition_block(self, l):
        # bringing down the channel
        l = self.conv_block(l, K.int_shape(l)[-1] // 2, kernel = (1,1), stride = (1,1))
        # downsampling the maps
        l = tf.keras.layers.AvgPool2D(2, strides = (2,2), padding = 'same')(l)
        return l

    def build_model(self):
        Input = tf.keras.layers.Input(self.input_shape)
        l = tf.keras.layers.Conv2D(64, (7,7), strides = (2,2), padding = 'same')(Input)
        l = tf.keras.layers.MaxPool2D(3, strides = (2,2), padding = 'same')(l)

        for layer in self.num_layers:
            # if self.num_layers.index(layer) == len(self.num_layers) - 1:
            #     # last block only contain dense block
            #     dense_block = self.dense_block(l, layer)
            #     continue
            dense_block = self.dense_block(l, layer)
            l = self.transition_block(dense_block)

        l = tf.keras.layers.GlobalAvgPool2D()(dense_block)
        Output = tf.keras.layers.Dense(self.n_cls, activation = 'softmax')(l)
        model = tf.keras.Model(Input, Output)
        return model

