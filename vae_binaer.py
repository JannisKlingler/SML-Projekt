import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import tensorflow as tf
import scipy.stats
from tensorflow import keras
from tensorflow.keras import layers

# Erstellen eines Trainingsdatensatzes und eines Testdatensatzes
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.where(x_train > 127.5, 1.0, 0.0).astype('float32')
x_test = np.where(x_test > 127.5, 1.0, 0.0).astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Modellparameter
latent_dim = 5
trainingsepochen = 20

# Encodermodell
encoder_inputs = keras.Input(shape=(784,))
dense = layers.Dense(500, activation="tanh")
x = dense(encoder_inputs)

# Reparametrisierungstrick
μ = layers.Dense(latent_dim)(x)
log_σ = layers.Dense(latent_dim)(x)


def reparam(args):
    μ, log_σ = args
    epsilon = K.random_normal(shape=(K.shape(μ)[0], latent_dim),
                              mean=0., stddev=1)
    return μ + K.exp(2 * log_σ) * epsilon


z = layers.Lambda(reparam)([μ, log_σ])

encoder = keras.Model(encoder_inputs, [μ, log_σ, z])

# Decodermodell
decoder = keras.Sequential([
    keras.Input(shape=(latent_dim,)),
    layers.Dense(500, activation="tanh"),
    layers.Dense(784, activation="sigmoid"),
]
)

# VAE
decoder_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, decoder_outputs)

# Trainingskriterium definieren
log_p_q = 784 * keras.losses.binary_crossentropy(encoder_inputs, decoder_outputs)
kl_div = K.sum(1 + 2 * log_σ - K.square(μ) - 2 * K.exp(log_σ), axis=-1)
vae_loss = .5 * K.mean(log_p_q - kl_div)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.fit(x_train, x_train,
        epochs=trainingsepochen,
        batch_size=100,
        validation_data=(x_test, x_test))


decoded_imgs = vae.predict(x_test)


n = 15  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
