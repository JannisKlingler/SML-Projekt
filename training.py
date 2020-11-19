import matplotlib.pyplot as plt
#from keras import backend as K
import numpy as np
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

import models
import lossfunctions


latent_dim = 5
trainingsepochen = 1


(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
data_dim = 784


x_train = np.where(x_train > 127.5, 1.0, 0.0).astype('float32')
x_train = np.reshape(x_train, (len(x_train), data_dim))


encoder = models.VAE_Dense_Encoder(data_dim, latent_dim, [500], "tanh")

decoder = models.Bernoulli_Dense_Decoder(data_dim, latent_dim, [500], "tanh")

vae = tf.keras.Model(encoder.inp, decoder(encoder(encoder.inp)[2]))


vae.add_loss(lossfunctions.Bernoulli_Loss(encoder, decoder))
vae.compile(optimizer='adam')



vae.fit(x_train, x_train,
        epochs=trainingsepochen,
        batch_size=100)


decoded_imgs = vae.predict(x_train)


n = 15
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
