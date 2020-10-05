import numpy as np
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from vae import VAE
from plots import plot_label_clusters, plot_latent, generate_images

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

    # images, labels, image_size = prepare_data()

    latent_dim = 128
    epochs = 20
    batch_size = 512

    # vae = VAE(image_size, latent_dim)
    # vae.compile(optimizer='adam')
    # vae.fit(images, epochs=epochs, batch_size=batch_size)

    # plot_latent(vae.encoder, vae.decoder, image_size, n=10)

    # plot_label_clusters(vae.encoder, vae.decoder, images, labels)

    from PIL import Image
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import os
    import numpy as np
    from glob import glob
    
    img_path = 'app/dataset/'

    filenames = np.array(glob(os.path.join(img_path, '*/*.jpg')))
    num_images = len(filenames)


    # for filename in filenames:
    #     try:
    #         im = Image.open(filename)
    #         im.verify()
    #     except:
    #         print(filename)
        

    img_size = 64

    vae = VAE(img_size, latent_dim)
    model = vae.model_build()

    # datagen = ImageDataGenerator(rescale=1./255)

    # data_flow = datagen.flow_from_directory(
    #     img_path,
    #     target_size=(img_size, img_size),
    #     shuffle=True,
    #     batch_size=batch_size,
    #     class_mode='input')

    # bestweights = keras.callbacks.ModelCheckpoint(filepath='app/saves/weights.h5', save_weights_only=True, verbose=1)
    # history = model.fit(data_flow, epochs=epochs, steps_per_epoch=(num_images // batch_size), callbacks=[bestweights])
    
    model.load_weights('app/saves/weights.h5')

    generate_images(vae.decoder, latent_dim, num=5)