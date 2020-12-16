import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import scipy as sp
import models
import lossfunctions
import picture

tf.random.set_seed(1)

# Aufgabe vorgeben. Mögliche Eingaben: 'MNIST', 'rotatingMNIST', 'GaussMNIST'
job = 'rectangles'

latent_dim = 4
epochs = 1

akt_fun = 'relu'

if job == 'MNIST':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.where(x_train > 127.5, 1.0, 0.0).astype('float32')
    x_test = np.where(x_test > 127.5, 1.0, 0.0).astype('float32')
    encoder = models.VAE_Conv_Encoder(latent_dim, akt_fun)
    decoder = models.Bernoulli_Conv_Decoder(latent_dim, akt_fun)
    loss = lossfunctions.Bernoulli_Loss(encoder, decoder, 1)
    #loss = lossfunctions.Trivial_Loss(encoder, decoder, 1)

elif job == 'GaussMNIST':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    encoder = models.VAE_Conv_Encoder(latent_dim, akt_fun)
    decoder = models.Gauss_Conv_Decoder(latent_dim, akt_fun)
    loss = lossfunctions.Gauss_Loss(encoder, decoder, 1)

elif job == 'rotatingMNIST':  # Passenden Dateipfad einfügen
    x_train = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train.npy')
    x_test = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test.npy')
    frames = 10
    #encoder = models.VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
    encoder = models.ODE_VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
    decoder = models.ODE_Bernoulli_ConvTime_Decoder(frames, latent_dim, akt_fun)
    loss = lossfunctions.Trivial_Loss(encoder, decoder, 10)

elif job == 'rectangles':  # Passenden Dateipfad einfügen
    x_train = picture.create_dataset(amount=100)
    x_test = picture.create_dataset(amount=10)
    frames = 10
    #encoder = models.VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
    encoder = models.ODE_VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
    decoder = models.ODE_Bernoulli_ConvTime_Decoder(frames, latent_dim, akt_fun)
    loss = lossfunctions.Trivial_Loss(encoder, decoder, 10)

else:
    raise Exception("That job does not exist")


vae = tf.keras.Model(encoder.inp, decoder(encoder(encoder.inp)[-1]))

vae.add_loss(loss)
vae.compile(optimizer='adam')

vae.fit(x_train, x_train,
        epochs=epochs,
        batch_size=100)
rec_imgs = vae.predict(x_test)[0]


n = 25
k = 0
if job == 'MNIST' or job == 'GaussMNIST':
    plt.figure(figsize=(20, 4))
    for i in np.random.randint(len(x_test), size=n):
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, k + 1 + n)
        plt.imshow(rec_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        k = k + 1

elif job == 'rotatingMNIST':
    fig, index = plt.figure(figsize=(10, 10)), np.random.randint(len(x_test), size=5)
    grid = ImageGrid(fig, 111,  nrows_ncols=(10, 10), axes_pad=0.1,)
    plot = [x_test[index[0]][:, :, j] for j in range(10)]
    for i in index:
        if k != 0:
            original = [x_test[i][:, :, j] for j in range(10)]
            plot = np.vstack((plot, original))
        reconst = [rec_imgs[i][:, :, j] for j in range(10)]
        plot = np.vstack((plot, reconst))
        k += 1
    for ax, im in zip(grid, plot):
        plt.gray()
        ax.imshow(im)
    plt.show()

elif job == 'rectangles':
    picture.show_dataset(x_test, rec_imgs)

# Bei Bedarf: Modell speichern und laden
#tf.keras.models.save_model(vae, 'C:/Pfad/vae')
#tf.keras.models.save_model(encoder, 'C:/Pfad/encoder')
#tf.keras.models.save_model(decoder, 'C:/Pfad/decoder')

#loaded_vae = tf.keras.models.load_model('C:/Pfad/vae')
#loaded_encoder = tf.keras.models.load_model('C:/Pfad/encoder')
#loaded_decoder = tf.keras.models.load_model('C:/Pfad/decoder')

#rec_imgs = loaded_vae.predict(x_train)
#encoded_imgs = loaded_encoder.predict(x_train)
#decoded_imgs = loaded_encoder.predict(x_train)
