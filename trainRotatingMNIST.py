import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import scipy as sp
import models
import lossfunctions
import picture

tf.random.set_seed(1)


latent_dim = 4
epochs = 1

akt_fun = 'relu'

x_train = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train.npy')
x_test = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test.npy')
frames = 10
#encoder = models.VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
encoder = models.ODE_VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
decoder = models.ODE_Bernoulli_ConvTime_Decoder(frames, latent_dim, akt_fun)
loss = lossfunctions.Trivial_Loss(encoder, decoder, 10)


vae = tf.keras.Model(encoder.inp, decoder(encoder(encoder.inp)[-1]))

vae.add_loss(loss)
vae.compile(optimizer='adam')

vae.fit(x_train, x_train,
        epochs=epochs,
        batch_size=100)
rec_imgs = vae.predict(x_test)[0]


n = 25
k = 0

fig, index = plt.figure(figsize=(10, 10)), np.random.randint(len(x_test), size=5)
grid = ImageGrid(fig, 111,  nrows_ncols=(10, 10), axes_pad=0.1,)
plot = [x_test[index[0]][:, :, j] for j in range(10)]
for i in index:
    if k != 0:
        original = [x_test[i][:, :, j] for j in range(10)]
        plot = np.vstack((plot, original))
    reconst = [rec_imgs[i][:, :, j] for j in range(10)]
    plot = np.vstack((plot, reconst))
    k += 1
for ax, im in zip(grid, plot):
    plt.gray()
    ax.imshow(im)
plt.show()
