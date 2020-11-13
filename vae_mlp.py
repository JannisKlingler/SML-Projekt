import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Modellparameter, Anzahl der latenten Dimensionen und Trainingsepochen.
latent_dim = 5
trainingsepochen = 1

# Erstellen eines Trainingsdatensatzes und eines Testdatensatzes
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# Binärisieren der Daten. Wir wählen als zugrundeliegende Verteilung eine
# (multivariate) Bernoulliverteilung, da der MNIST-Datensatz annährend aus 0,1 Werten
# besteht und das approximieren stetiger Werte z.B. mit einer multivariaten Gaußverteilung
# bei solchen Datensätzen oft instabil ist.
x_train = np.where(x_train > 127.5, 1.0, 0).astype('float32')
x_test = np.where(x_test > 127.5, 1.0, 0).astype('float32')
x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))

# Encodermodell mit einem hidden layer mit 500 Neuronen und Aktivierungsfunktion f(x)=tanh(x)
encoder_input = keras.Input(shape=(784,))
x = layers.Dense(500, activation="tanh")(encoder_input)

# Reparametrisierungstrick. Siehe im Paper Auto-Encoding Variational Bayes S.4
# Es wird aus Gründen der Stabilität log(σ) statt σ approximiert.
μ = layers.Dense(latent_dim, name="mu")(x)
log_σ = layers.Dense(latent_dim, name="log_sig")(x)


def reparam(args):
    μ, log_σ = args
    epsilon = K.random_normal(shape=(K.shape(μ)[0], latent_dim),
                              mean=0., stddev=1)
    return μ + K.exp(log_σ) * epsilon


z = layers.Lambda(reparam)([μ, log_σ])
encoder = keras.Model(encoder_input, [μ, log_σ, z], name="Encoder")
encoder.summary()  # Zusammenfassung des Encodermodells
# Decodermodell mit einem hidden layer mit 500 Neuronen und Aktivierungsfunktion f(x)=tanh(x)
# Sigmoidaktivierungsfunktion im Outputlayer, da der Output die
# Parameter einer multivar. Benoulliverteilung sind und daher in [0,1] fallen müssen.
decoder = keras.Sequential([
    keras.Input(shape=(latent_dim,)),
    layers.Dense(500, activation="tanh"),
    layers.Dense(784, activation="sigmoid"), ], name="Decoder")
decoder.summary()  # Zusammenfassung des Decodermodells
# VAE
decoder_output = decoder(encoder(encoder_input)[2])
vae = keras.Model(encoder_input, decoder_output)

# Trainingskriterium definieren. Dies ist die ELBO, siehe Auto-Encoding Variational Bayes S.5,11
# logp(x|z) ist für die Bernoulliverteilung die Kreuzentropie.
log_p_xz = 784 * keras.losses.binary_crossentropy(encoder_input, decoder_output)
kl_div = K.sum(1 + 2 * log_σ - K.square(μ) - 2 * K.exp(log_σ), axis=-1)
elbo = .5 * K.mean(log_p_xz - kl_div)
vae.add_loss(elbo)
vae.compile(optimizer='adam')


# Training des Netzwerkes
vae.fit(x_train, x_train,
        epochs=trainingsepochen,
        batch_size=100,
        validation_data=(x_test, x_test))


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
