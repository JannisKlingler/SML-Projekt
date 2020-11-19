import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import models
import lossfunctions

tf.random.set_seed(0)

# Hyperparameter. latent_dim = 10 und epochs = 100 liefert gute Ergebnisse
latent_dim = 10
epochs = 100
encoder_struc = [784, 500, 500, latent_dim]
decoder_struc = [latent_dim, 500, 500, 784]
dropout = 0.1  # Auf 0 setzen, falls nicht gewünscht
akt_fun = "tanh"

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = np.where(x_train > 127.5, 1.0, 0.0).astype('float32')
x_train = np.reshape(x_train, (len(x_train), encoder_struc[0]))

encoder = models.VAE_Dense_Encoder(encoder_struc, dropout, akt_fun)
decoder = models.Bernoulli_Dense_Decoder(decoder_struc, dropout, akt_fun)

vae = tf.keras.Model(encoder.inp, decoder(encoder(encoder.inp)[2]))


vae.add_loss(lossfunctions.Bernoulli_Loss(encoder, decoder))
vae.compile(optimizer='adam')

vae.fit(x_train, x_train,
        epochs=epochs,
        batch_size=100)

decoded_imgs = vae.predict(x_train)


n = 25
k = 0
plt.figure(figsize=(20, 4))
for i in np.random.randint(len(x_train), size=n):

    ax = plt.subplot(2, n, k + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, k + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    k = k + 1
plt.show()
