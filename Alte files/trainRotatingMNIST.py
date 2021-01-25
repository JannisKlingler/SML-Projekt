import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import scipy as sp
import models
import lossfunctions

tf.random.set_seed(1)


latent_dim = 20
epochs = 1
frames = 10
DataSize = -1 #Für ganzen MNIST-Datenstaz wähle -1
akt_fun = 'relu'


try:
    x_train = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train.npy')
    x_test = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test.npy')
except:
    print('Dataset is being generated. This may take a few minutes.')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[0:DataSize]
    x_test = x_test[0:DataSize]
    x_train_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_train))
    x_test_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_test))
    for j in range(len(x_test_rot)):
        for i in np.random.choice(range(3, 10), 3, replace=False):
            x_test_rot[j][i] = np.zeros((28, 28))
    x_train = np.transpose(np.array(x_train_rot), [0, 2, 3, 1])
    x_test = np.transpose(np.array(x_test_rot), [0, 2, 3, 1])
    try:
        np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train', x_train)
        np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test', x_test)
    except:
        print('could not save Dataset')
    print('Dataset generated')


#encoder = models.VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
encoder = models.VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
decoder = models.Bernoulli_ConvTime_Decoder(frames, latent_dim, akt_fun)
loss = lossfunctions.Bernoulli_Loss(encoder, decoder, 10)

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


# plt.savefig('C:/Users/Admin/Desktop/Ergebnisse/ConvTime.png')

plt.show()
