import numpy as np
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from vae import VAE
from plots import plot_label_clusters, plot_latent

def config_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

def prepare_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    images = np.concatenate([x_train, x_test], axis=0)
    labels = np.concatenate([y_train, y_test], axis=0)

    images = np.expand_dims(images, -1).astype("float32") / 255

    return images, labels, images.shape[1]

if __name__ == '__main__':
    config_gpu()

    images, labels, image_size = prepare_data()

    latent_dim = 2
    epochs = 50
    batch_size = 2048

    vae = VAE(image_size, latent_dim)
    vae.compile(optimizer='adam')
    vae.fit(images, epochs=epochs, batch_size=batch_size)

    plot_latent(vae.encoder, vae.decoder, image_size, n=10)

    plot_label_clusters(vae.encoder, vae.decoder, images, labels)