import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy as sp

import models
import lossfunctions

tf.random.set_seed(0)

# Aufgabe vorgeben. Mögliche eingaben: 'MNIST', 'rotatingMNIST'
job = 'rotatingMNIST'

# Hyperparameter. latent_dim = 10 und epochs = 100 liefert gute Ergebnisse
latent_dim = 20
epochs = 30
#encoder_struc = [784, 500, 500, latent_dim]
#decoder_struc = [latent_dim, 500, 500, 784]
dropout = 0.1  # Auf 0 setzen, falls nicht gewünscht
akt_fun = 'tanh'

if job == 'MNIST':
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = np.where(x_train > 127.5, 1.0, 0.0).astype('float32')
    x_train = np.reshape(x_train, (len(x_train), 784))

    encoder = models.VAE_Conv_Encoder(latent_dim, akt_fun)
    decoder = models.Bernoulli_Conv_Decoder(latent_dim, akt_fun)
    loss = lossfunctions.Bernoulli_Loss(encoder, decoder)

elif job == 'rotatingMNIST':
    #x_train = np.load('rotatingMNIST_test.npy')
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[0:20000]
    x_train = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
        b, (i+1) * 360/10, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(10))), x_train))

    x_train[0][3] = np.zeros((28, 28))
    x_train = np.transpose(x_train, [0, 2, 3, 1])

    encoder = models.VAE_ConvTime_Encoder(latent_dim, akt_fun)
    decoder = models.Bernoulli_ConvTime_Decoder(latent_dim, akt_fun)
    loss = lossfunctions.BernoulliConv_Loss(encoder, decoder)
else:
    raise Exception("That job does not exist")


vae = tf.keras.Model(encoder.inp, decoder(encoder(encoder.inp)[2]))

vae.add_loss(loss)
vae.compile(optimizer='adam')

vae.fit(x_train, x_train,
        epochs=epochs,
        batch_size=100,
        verbose=2)

rec_imgs = vae.predict(x_train)


n = 20
k = 0
if job == 'MNIST':
    plt.figure(figsize=(20, 4))
    for i in np.random.randint(len(x_train), size=n):
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(x_train[i].reshape(28, 28))
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
    rec_imgs = np.transpose(rec_imgs, [0, 3, 1, 2])
    x_train = np.transpose(x_train, [0, 3, 1, 2])
    plt.figure(figsize=(20, 4))
    j = 0
    for i in range(20):
        ax = plt.subplot(2, n, k + 1)
        plt.imshow(x_train[k//10][i % 10])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, k + 1 + n)
        plt.imshow(rec_imgs[k//10][i % 10])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        k = k + 1
    #plt.figure(figsize=(20, 20))
    # for j in range(1):
        # for i in range(10):
        #ax = plt.subplot(10, 10, j*10+(i+1))
        # plt.imshow(rec_imgs[j][i])
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
plt.show()

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
