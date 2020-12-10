# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
import scipy.io

# https://cs.nyu.edu/~roweis/data.html Datensatz "Frey Face" Downloadlink (1MB)

# %%
latent_dim = 10
trainingsepochen = 100

mat_data = scipy.io.loadmat('C:/Users/Admin/Desktop/Python/frey_rawface.mat')
x_train = mat_data['ff'].T.reshape(-1, 28, 20, 1)
x_train = x_train.astype('float32') / 255.0
print(f"Größe Datensatz: {len(x_train)}")


encoder_input = keras.Input(shape=(28, 20, 1))
x = layers.Conv2D(32, (3, 3), padding="same", activation='relu')(encoder_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 2), activation='relu')(x)
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
x = layers.Dense(4 * 3 * 128, activation='relu')(x)
x = layers.Reshape((4, 3, 128))(x)
x = layers.Conv2DTranspose(64, (3, 2), activation='relu')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.Conv2DTranspose(1, (3, 3), padding="same", activation='sigmoid')(x)

decoder = keras.Model(decoder_input, x, name="Decoder")
decoder.summary()

decoder_output = decoder(encoder(encoder_input)[2])
vae = keras.Model(encoder_input, decoder_output)

log_p_xz = 784. * K.mean(keras.losses.binary_crossentropy(encoder_input, decoder_output))
kl_div = .5 * K.sum(1. + 2. * log_σ - K.square(μ) - 2. * K.exp(log_σ), axis=-1)
elbo = (log_p_xz - kl_div)
vae.add_loss(elbo)
vae.compile(optimizer='adam')

# %%
vae.fit(x_train,
        epochs=trainingsepochen,
        batch_size=100,
        verbose=2)

rec_imgs = vae.predict(x_train)

# %%
n = 25
k = 0
plt.figure(figsize=(20, 4))
for i in np.random.randint(len(x_train), size=n):
    ax = plt.subplot(2, n, k + 1)
    plt.imshow(x_train[i].reshape(28, 20))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, k + 1 + n)
    plt.imshow(rec_imgs[i].reshape(28, 20))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    k = k + 1
plt.show()

# %%

if latent_dim == 2:
    n = 15
    encoded_imgs = encoder.predict(x_train)

    x_1 = np.min(encoded_imgs[2][:, 0]) - np.min(encoded_imgs[2][:, 0]) / 4
    x_2 = np.max(encoded_imgs[2][:, 0]) - np.max(encoded_imgs[2][:, 0]) / 4
    y_1 = np.min(encoded_imgs[2][:, 1]) - np.min(encoded_imgs[2][:, 1]) / 4

    figure = np.zeros((28 * n, 20 * n))
    for i, yi in enumerate(np.linspace(x_1, x_2, n)):
        for j, xi in enumerate(np.linspace(y_1, y_1 + abs(x_1 - x_2), n)):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(28, 20)
            figure[i * 28: (i + 1) * 28, j * 20: (j + 1) * 20] = digit
    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='gray')
    plt.show()
