import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = np.where(x_train > 127.5, 1.0, 0.0).reshape(60000, 28, 28, 1)

latent_dim = 10
epochs = 15
batch_size = 100

encoder_input = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(encoder_input)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(latent_dim + latent_dim)(x)

μ, log_σ = tf.split(x, num_or_size_splits=2, axis=1)

z = tf.keras.layers.Lambda(lambda arg: arg[0] + tf.exp(arg[1]) * tf.random.normal(
    shape=(batch_size, latent_dim)))([μ, log_σ])

decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(128, activation='relu')(decoder_input)
x = tf.keras.layers.Dense(4 * 4 * 128, activation='relu')(x)
x = tf.keras.layers.Reshape((4, 4, 128))(x)
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu')(x)
x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
decoder_output = tf.keras.layers.Conv2DTranspose(1, (4, 4), padding='same', activation='sigmoid')(x)


encoder = tf.keras.Model(encoder_input, [μ, log_σ, z])
encoder.summary()
decoder = tf.keras.Model(decoder_input, decoder_output)
decoder.summary()
vae = tf.keras.Model(encoder_input, decoder(encoder(encoder_input)[2]))


def VAE_Loss(x, x_rec):
    μ, log_σ, z = encoder(x)
    x_rec = decoder(z)
    log_p_xz = tf.reduce_sum(x * tf.math.log(1e-8 + x_rec) + (1. - x)
                             * tf.math.log(1. - 1e-8 - x_rec))
    kl_div = - 0.5 * tf.reduce_sum(2 * log_σ + 1 - μ ** 2 - 2 * tf.exp(log_σ))
    return (- log_p_xz + kl_div) / batch_size


vae.compile(optimizer='adam',
            loss=VAE_Loss,
            metrics=[tf.keras.metrics.MeanSquaredError(name='MSE')])

vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

encoded_imgs = encoder.predict(x_train, batch_size=100)
decoded_imgs = vae.predict(x_train, batch_size=100)


plt.figure(figsize=(20, 4))
for k, i in enumerate(np.random.randint(len(x_train), size=15)):
    ax = plt.subplot(2, 15, k + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 15, k + 1 + 15)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

if latent_dim == 2:
    plt.figure(figsize=(10, 8))
    plt.scatter(encoded_imgs[2][:, 0], encoded_imgs[2][:, 1], s=1, c=y_train,
                cmap='tab10', marker='o')
    plt.colorbar()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.show()

    figure = np.zeros((28 * 25, 28 * 25))
    for i, yi in enumerate(np.linspace(-2, 2, 25)):
        for j, xi in enumerate(np.linspace(-2, 2, 25)):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(28, 28)
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
    plt.imshow(figure, cmap='gray')
    plt.show()

if latent_dim == 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_imgs[2][:, 0][0:30000], encoded_imgs[2][:, 1][0:30000],
               encoded_imgs[2][:, 2][0:30000], s=1, c=y_train[0:30000], cmap='tab10', marker='o')
    plt.show()
    plt.grid()

    print('Generate animation of latent space')
    frames = 45
    fig = plt.figure(figsize=(10, 10))
    ims = []
    for k in range(frames):
        figure = np.zeros((28 * 25, 28 * 25))
        for i, yi in enumerate(np.linspace(-2, 2, 25)):
            for j, xi in enumerate(np.linspace(-2, 2, 25)):
                z_sample = np.array([[xi, yi, -2 + k * 4 / frames]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(28, 28)
                figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
        print('Frame {} of {}'.format(k+1, frames))
        im = plt.imshow(figure, cmap='gray', animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
#    ani.save('latenter_raum.gif', writer='imagemagick', fps=30)
    plt.show()
