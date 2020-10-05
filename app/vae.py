import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers

class VAE():
    def __init__(self, image_size, latent_dim):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.encoder = self.encoder_build()
        self.decoder = self.decoder_build()

    def encoder_build(self):
        self.encoder_inputs = keras.Input(shape=(self.image_size, self.image_size, 3))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(self.encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        self.encoder_last_shape = x.shape[1:]

        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(self.encoder_inputs, [z_mean, z_log_var, z])

        return encoder

    def decoder_build(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(np.prod(self.encoder_last_shape), activation="relu")(latent_inputs)
        x = layers.Reshape(self.encoder_last_shape)(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs)

        return decoder

    def model_build(self):
        z_mean, z_log_var, z = self.encoder(self.encoder_inputs)
        outputs = self.decoder(z)
        model = Model(self.encoder_inputs, outputs)

        reconstruction_loss = binary_crossentropy(self.encoder_inputs, outputs)
        reconstruction_loss *= self.image_size * self.image_size

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
        model.add_loss(vae_loss)
        model.compile(optimizer='adam')

        return model


    # def call(self, data):
    #     if isinstance(data, tuple):
    #         data = data[0]
    #     with tf.GradientTape() as tape:
    #         z_mean, z_log_var, z = self.encoder(data)
    #         reconstruction = self.decoder(z)
    #         reconstruction_loss = tf.reduce_mean(
    #             keras.losses.binary_crossentropy(data, reconstruction)
    #         )
    #         reconstruction_loss *= self.image_size * self.image_size
    #         kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    #         kl_loss = tf.reduce_mean(kl_loss)
    #         kl_loss *= -0.5
    #         total_loss = reconstruction_loss + kl_loss
    #     grads = tape.gradient(total_loss, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #     return {
    #         "loss": total_loss,
    #         "reconstruction_loss": reconstruction_loss,
    #         "kl_loss": kl_loss,
    #     }

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, latent_dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon