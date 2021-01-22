import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Flatten, Dense, Reshape

class VAE():
    def __init__(self, image_size, latent_dim):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.units_init = 64
        self.num_conv_layers = 4
        self.encoder = self.encoder_build()
        self.decoder = self.decoder_build()

    def encoder_build(self):
        self.encoder_inputs = keras.Input(shape=(self.image_size, self.image_size, 3))
        x = self.encoder_inputs

        for i in range(self.num_conv_layers):
            num_units = self.units_init * (2**i)
            
            x = Conv2D(num_units, 3, strides=2, padding="same")(x)
            x = ReLU()(x)

        self.encoder_last_shape = x.shape[1:]

        x = Flatten()(x)
        x = Dense(self.latent_dim * 2)(x)
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(self.encoder_inputs, [z_mean, z_log_var, z])

        return encoder

    def decoder_build(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = Dense(np.prod(self.encoder_last_shape))(latent_inputs)
        x = Reshape(self.encoder_last_shape)(x)

        for i in range(self.num_conv_layers):
            exp = -1 * (i + 1 - self.num_conv_layers)
            num_units = self.units_init * (2**exp)

            x = Conv2DTranspose(num_units, 3, strides=2, padding="same")(x)
            x = ReLU()(x)

        decoder_outputs = Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs)

        return decoder

    def model_build(self):
        z_mean, z_log_var, z = self.encoder(self.encoder_inputs)
        outputs = self.decoder(z)
        model = Model(self.encoder_inputs, outputs)

        reconstruction_loss = binary_crossentropy(self.encoder_inputs, outputs)
        reconstruction_loss *= self.image_size * self.image_size * 3

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
        model.add_loss(vae_loss)
        model.compile(optimizer='adam')

        return model

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, latent_dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon