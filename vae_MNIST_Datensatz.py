import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import tensorflow as tf
import scipy.stats
from tensorflow import keras
from tensorflow.keras import layers
import gzip
import pickle
import sys

# Modellparameter, Anzahl der latenten Dimensionen und Trainingsepochen.
latent_dim = 5
trainingsepochen = 10

# Erstellen eines Trainingsdatensatzes und eines Testdatensatzes
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()
(x_train, _), (x_test, _) = data

# Binärisieren der Daten. Wir wählen als zugrundeliegende Verteilung eine
# (multivariate) Bernoulliverteilung, da der MNIST-Datensatz annährend aus 0,1 Werten
# besteht und das approximieren stetiger Werte z.B. mit einer multivariaten Gaußverteilung
# bei solchen Datensätzen oft instabil ist.
x_train = np.where(x_train > 127.5, 1.0, 0.0).astype('float32')
x_test = np.where(x_test > 127.5, 1.0, 0.0).astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Encodermodell mit einem hidden layer mit 500 Neuronen und Aktivierungsfunktion f(x)=tanh(x)
encoder_inputs = keras.Input(shape=(784,))
dense = layers.Dense(500, activation="tanh")
x = dense(encoder_inputs)

# Reparametrisierungstrick. Siehe im Paper Auto-Encoding Variational Bayes S.4
# Es wird aus Gründen der Stabilität log(σ) statt σ approximiert.
μ = layers.Dense(latent_dim)(x)
log_σ = layers.Dense(latent_dim)(x)


def reparam(args):
    μ, log_σ = args
    epsilon = K.random_normal(shape=(K.shape(μ)[0], latent_dim),
                              mean=0., stddev=1)
    return μ + K.exp(log_σ) * epsilon


z = layers.Lambda(reparam)([μ, log_σ])

encoder = keras.Model(encoder_inputs, [μ, log_σ, z])

# Decodermodell mit einem hidden layer mit 500 Neuronen und Aktivierungsfunktion f(x)=tanh(x)
# Sigmoidaktivierungsfunktion im Outputlayer, da der Output die
# Parameter einer multivar. Benoulliverteilung sind und daher in [0,1] fallen müssen.
decoder = keras.Sequential([
    keras.Input(shape=(latent_dim,)),
    layers.Dense(500, activation="tanh"),
    layers.Dense(784, activation="sigmoid"),
]
)

# VAE
decoder_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, decoder_outputs)

# Trainingskriterium definieren. Dies ist die ELBO, siehe Auto-Encoding Variational Bayes S.5,11
# logp(x|z) ist für die Bernoulliverteilung die Kreuzentropie.
log_p_xz = 784 * keras.losses.binary_crossentropy(encoder_inputs, decoder_outputs)
kl_div = .5 * K.sum(1 + 2 * log_σ - K.square(μ) - 2 * K.exp(log_σ), axis=-1)
elbo = K.mean(log_p_xz - kl_div)
vae.add_loss(elbo)
vae.compile(optimizer='adam')

# Training des Netzwerkes
vae.fit(x_train, x_train,
        epochs=trainingsepochen,
        batch_size=100,
        validation_data=(x_test, x_test), verbose=1)


decoded_imgs = vae.predict(x_test)

# Visualisierung der Ergebnisse
n = 15  # Wieviele Bilder angezeigt werden sollen
k = 0
plt.figure(figsize=(20, 4))
for i in np.random.randint(len(x_test), size=n):
    # Original
    ax = plt.subplot(2, n, k + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Rekonstruktion. Die Grauwerte stellen die encodierte Verteilung dar.
    # jedes Pixel ist Parameter einer Bernoulliverteilung aus welcher das Ergebnis gezogen werden kann.
    ax = plt.subplot(2, n, k + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    k = k + 1
plt.show()
