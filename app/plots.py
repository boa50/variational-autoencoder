import numpy as np
import matplotlib.pyplot as plt

def generate_images(decoder, latent_dim, num=10):
    reconst_images = decoder.predict(np.random.normal(0,1,size=(num,latent_dim)))

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(num):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, num, i+1)
        sub.axis('off')        
        sub.imshow(img)
    
    plt.show()

def generate_reconstructions(model, data_flow):
    example_batch = next(data_flow)
    example_batch = example_batch[0]
    images = example_batch[:10]

    n_to_show = images.shape[0]
    reconst_images = model.predict(images)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')        
        sub.imshow(img)

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
        sub.axis('off')
        sub.imshow(img)

    plt.show()