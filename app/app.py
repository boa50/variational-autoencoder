import numpy as np
import os
from glob import glob

from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from vae import VAE
from plots import generate_images, generate_reconstructions

def config_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)

def check_corrupted_images(filenames):
    from PIL import Image

    for filename in filenames:
        try:
            im = Image.open(filename)
            im.verify()
        except:
            print(filename)


def prepare_data(img_size, batch_size, check_corrupted=False):
    img_path = 'app/dataset/'

    filenames = np.array(glob(os.path.join(img_path, '*/*.jpg')))
    num_images = len(filenames)

    if check_corrupted:
        check_corrupted_images(filenames)

    datagen = ImageDataGenerator(rescale=1./255)

    data_flow = datagen.flow_from_directory(
        img_path,
        target_size=(img_size, img_size),
        shuffle=True,
        batch_size=batch_size,
        class_mode='input')

    return data_flow, num_images

if __name__ == '__main__':
    config_gpu()

    latent_dim = 256
    epochs = 10000
    batch_size = 128
    img_size = 64

    vae = VAE(img_size, latent_dim)
    model = vae.model_build()

    data_flow, num_images = prepare_data(img_size, batch_size)

    steps_per_epoch = num_images // batch_size
    save_period = 50

    # weights_cb = keras.callbacks.ModelCheckpoint(
    #     filepath='app/saves/weights_l256_64_4_{epoch:02d}.h5', 
    #     save_weights_only=True,
    #     save_freq=int(save_period * steps_per_epoch),
    #     verbose=0)

    # model.fit(
    #     data_flow, 
    #     epochs=epochs, 
    #     steps_per_epoch=steps_per_epoch, 
    #     callbacks=[weights_cb], 
    #     verbose=1)

    model.load_weights('app/saves/weights_l256_64_4_1150.h5')

    generate_reconstructions(model, data_flow)
    generate_images(vae.decoder, latent_dim, num=20)