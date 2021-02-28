from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose,Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
import numpy as np
import os
import pickle


class Autoencoder:
    def __init__(self, input_shape,
                 conv_filters,
                 conv_kernel,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.shape_before_bottleneck = None
        self._model_input = None

        self.encoder = None
        self.decoder = None
        self.model = None

        self.num_conv_layers = len(conv_filters)
        self._build()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def compile(self, learning_rate=0.001):
        optimizer = Adam(lr=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer, mse_loss)

    def train(self, x_train, batch_size, epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       shuffle=True)

    def reconstruct(self,images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images,latent_representations

    def _build_encoder(self):
        # creates all convolutionals blocks in encoder
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name='encoder')

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self.num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        # adds convolutional block to a graph of layers, consisting of conv2d = relu + batchnorm
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernel[layer_index],
            strides=self.conv_strides[layer_index],
            padding='same',
            name='encoder_conv_layer_{}'.format(layer_number)
        )
        x = conv_layer(x)
        x = ReLU(name='encoder_relu_{}'.format(layer_number))(x)
        x = BatchNormalization(name='encoder_batchnorm_{}'.format(layer_number))(x)
        return x

    def _add_bottleneck(self, x):
        # flatten data and add bottleneck(Dense layer)
        # save the input size for the purpose of the decoder
        self.shape_before_bottleneck = K.int_shape(x)[1:]  # [2,7,7,7] e.g. shape
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name='encoder_output')(x)
        return x

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layer = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layer)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self.shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name='decoder_dense')(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self.shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        # add conv block
        for layer_index in reversed(range(1, self.num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self.num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernel[layer_index],
            strides=self.conv_strides[layer_index],
            padding='same',
            name='decoder_conv_transpose_layer_{}'.format(layer_num)
        )
        x = conv_transpose_layer(x)
        x = ReLU(name='decoder_relu_{}'.format(layer_num))(x)
        x = BatchNormalization(name='decoder_batchnorm_{}'.format(layer_num))(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernel[0],
            strides=self.conv_strides[0],
            padding='same',
            name='decoder_conv_transpose_layer_{}'.format(self.num_conv_layers)
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name='sigmoid_layer')(x)
        return output_layer

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name='autoencoder')

