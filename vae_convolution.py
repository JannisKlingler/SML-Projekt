import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

latent_dim = 10
trainingsepochen = 60

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.where(x_train > 127.5, 1.0, 0.0).astype('float32')
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))


encoder_input = keras.Input(shape=(28, 28, 1))
encoder_input = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), padding="same", activation='relu')(encoder_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

μ = layers.Dense(latent_dim, name="mu")(x)
log_σ = layers.Dense(latent_dim, name="log_sig")(x)


def reparam(args):
    μ, log_σ = args
    epsilon = K.random_normal(shape=(K.shape(μ)[0], latent_dim), mean=0., stddev=1)
    return μ + K.exp(log_σ) * epsilon


z = layers.Lambda(reparam)([μ, log_σ])
encoder = keras.Model(encoder_input, [μ, log_σ, z], name="Encoder")
encoder.summary()

decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(decoder_input)
x = layers.Dense(4 * 4 * 128, activation='relu')(x)
x = layers.Reshape((4, 4, 128))(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.Conv2DTranspose(1, (4, 4), padding="same", activation='sigmoid')(x)

decoder = keras.Model(decoder_input, x, name="Decoder")
decoder.summary()

decoder_output = decoder(encoder(encoder_input)[2])
vae = keras.Model(encoder_input, decoder_output)

log_p_xz = 784. * K.mean(keras.losses.binary_crossentropy(encoder_input, decoder_output))
kl_div = .5 * K.sum(1. + 2. * log_σ - K.square(μ) - 2. * K.exp(log_σ), axis=-1)
elbo = (log_p_xz - kl_div)
vae.add_loss(elbo)
vae.compile(optimizer='adam')


vae.fit(x_train, x_train,
        epochs=trainingsepochen,
        batch_size=100,
        verbose=2)
