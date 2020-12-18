import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import scipy as sp
import models
import lossfunctions


tf.random.set_seed(1)


latent_dim = 4
epochs = 1

akt_fun = 'relu'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.where(x_train > 127.5, 1.0, 0.0).astype('float32')
x_test = np.where(x_test > 127.5, 1.0, 0.0).astype('float32')
encoder = models.VAE_Conv_Encoder(latent_dim, akt_fun)
decoder = models.Bernoulli_Conv_Decoder(latent_dim, akt_fun)
loss = lossfunctions.Bernoulli_Loss(encoder, decoder, 1)


vae = tf.keras.Model(encoder.inp, decoder(encoder(encoder.inp)[-1]))

vae.add_loss(loss)
vae.compile(optimizer='adam')

vae.fit(x_train, x_train,
        epochs=epochs,
        batch_size=100)
rec_imgs = vae.predict(x_test)[0]


n = 25
k = 0

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
plt.show()
