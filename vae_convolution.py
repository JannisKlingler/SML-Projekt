import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

latent_dim = 5
trainingsepochen = 10

# Dieses Modell hat einfach alles! Convolution, Pooling, Dropout, Fully Connected Layer, uvm!
# Leider benötigt das Training auch dementsprechend länger.
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = np.where(x_train > 127.5, 1.0, 0).astype('float32')
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))


encoder_input = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

μ = layers.Dense(latent_dim, name="mu")(x)
log_σ = layers.Dense(latent_dim, name="log_sig")(x)


def reparam(args):
    μ, log_σ = args
    epsilon = K.random_normal(shape=(100, latent_dim), mean=0., stddev=1)
    return μ + K.exp(log_σ) * epsilon


z = layers.Lambda(reparam)([μ, log_σ])
encoder = keras.Model(encoder_input, [μ, log_σ, z], name="Encoder")
encoder.summary()

decoder = keras.Sequential([
    layers.InputLayer(input_shape=(latent_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(14 * 14 * 32, activation='relu'),
    layers.Reshape((14, 14, 32)),
    layers.Dropout(0.25),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'), ], name="Decoder")
decoder.summary()

decoder_output = decoder(encoder(encoder_input)[2])
vae = keras.Model(encoder_input, decoder_output)

log_p_xz = 784 * K.mean(keras.losses.binary_crossentropy(encoder_input, decoder_output))
kl_div = K.sum(1 + 2 * log_σ - K.square(μ) - 2 * K.exp(log_σ), axis=-1)
elbo = .5 * (log_p_xz - kl_div)
vae.add_loss(elbo)
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
