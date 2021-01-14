import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy as sp
from keras import backend as K
import SDE_Tools
import AE_Tools
tfd = tfp.distributions
# import datasets as data

'''
# Needed for gpu support on some machines
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
'''

########################################################
# %% hyperparameter
epochs = 1
latent_dim = 5  # Dimensionality for latent variables. 20-30 works fine.
batch_size = 10  # ≥100 as suggested by Kingma in Autoencoding Variational Bayes.
train_size = 500  # Data points in train set. Choose accordingly to dataset size.
test_size = 100 # Data points in test set. Choose accordingly to dataset size.
batches = int(train_size / batch_size)
frames = 1  # Number of images in every datapoint. Choose accordingly to dataset size.
armortized_len = 3  # Sequence size seen by velocity encoder network.
act = 'relu'  # Activation function 'tanh' is used in odenet.
T = 1  # number of seconds of the video
fps = T/frames
n = 1
pictureWidth = 28
pictureHeight = 28
pictureColors = 1
M = 1
complexity = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train[0:train_size]
x_train = np.where(x_train > 127.5, 1.0, 0.0).astype('float32')
x_test = x_test[0:test_size]
x_test = np.where(x_test > 127.5, 1.0, 0.0).astype('float32')

print('train starting shape:',x_train.shape)

# Dim: train_size x frames x pictureWidth x pictureHeight x pictureColors
x_train = np.transpose(np.array([[x_train]]), (2,0,3,4,1))
print('train-shape:', x_train.shape)


# Dim: test_size x frames x pictureWidth x pictureHeight x pictureColors
x_test = np.transpose(np.array([[x_test]]), (2,0,3,4,1))


########################################################
# Datensatz für Encoder erstellen

#Dim: train_size*(frames-M+1) x pictureWidth x pictureHeight x (M*pictureColors)
x_train_longlist = AE_Tools.make_training_data(x_train, train_size, frames, M)
print('new_train_shape:',x_train_longlist.shape)



######################################

P_dec = AE_Tools.FramewiseDecoder(latent_dim, pictureWidth, pictureHeight, pictureColors, act, complexity=complexity)
P_enc = AE_Tools.LocalEncoder(latent_dim, M, pictureWidth, pictureHeight, pictureColors, act, complexity=complexity, variational=True)
#AE = AE_Tools.SimpleAutoencoder(P_enc, P_dec)
VAE = AE_Tools.VariationalAutoencoder(P_enc, P_dec)

#ms_Net = SDE_Tools.mu_sig_Net(latent_dim, n, act, 10)
#reconstructor = SDE_Tools.make_Tensorwise_Reconstructor(latent_dim*pictureColors, n, T, frames, ms_Net, batch_size)

loss = AE_Tools.make_binary_crossentropy_rec_loss(M)

#AE.compile(optimizer='adam', loss=loss)
#AE.fit(x_train_longlist, x_train_longlist[:,:,:,:,:], epochs=epochs, batch_size=batch_size, shuffle=False)



VAE.compile(optimizer='adam', loss=loss)
VAE.fit(x_train_longlist, x_train_longlist[:,:,:,:,:], epochs=epochs, batch_size=batch_size, shuffle=False)



######################################

#x_test = data.create_dataset(dataset_size=100, frames=10, picture_size=28, object_number=3)
k = 0
x_org = x_train_longlist[0:10,0,:,:,0]
print('x_org:',x_org.shape)
enc_imgs = list(map(lambda i: P_enc(x_train_longlist[i,:,:,:,:]), range(10)))
enc_imgs = tf.stack(enc_imgs, axis=0)[:,:,-1,:]
#enc_imgs = P_enc(x_train_longlist[:,0,:,:,:])
print(enc_imgs.shape)

rec_imgs = VAE.predict(x_train_longlist[0:10,:,:,:,:])

print('rec_imgs:',rec_imgs.shape)

fig, axs = plt.subplots(3, 5)
axs[0, 0].set_title('latent_dim:{}'.format(latent_dim))
axs[0, 1].set_title('complexity:{}'.format(complexity))
axs[0, 2].set_title('train_size:{}'.format(train_size))
axs[0, 3].set_title('batch_size:{}'.format(batch_size))
for i in range(5):
    axs[0, i].imshow(x_org[i,:,:], cmap='gray', vmin=0, vmax=1)
    axs[1, i].imshow(rec_imgs[i,0,:,:,0], cmap='gray', vmin=0, vmax=1)
    axs[2, i].plot(np.linspace(1,latent_dim,latent_dim),enc_imgs[i,0,:],'o')

plt.show()
