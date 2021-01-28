#import datasets as data
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import SDE_Tools
import AE_Tools


latent_dim = 10  # 5-30 sollte gut gehen
batch_size = 50  # eher klein halten, unter 100 falls möglich, 50 klappt gut
train_size = 60000
test_size = 10  # wird hier noch nicht gebraucht
frames = 20  # Number of images in every datapoint. Choose accordingly to dataset size.
act_CNN = 'relu'  # Activation function 'tanh' is used in odenet.
act_ms_Net = 'tanh'
Time = 50  # number of seconds of the video
SDE_Time = 250
fps = Time/frames
n = 1
pictureWidth = 28
pictureHeight = 28
pictureColors = 1
M = 2  # für 2 ist es ein 2-SDE-VAE, M sollte nie größer als frames//2 sein
CNN_complexity = 20  # wird zur zeit garnicht verwenden
SDE_Net_complexity = 8*latent_dim  # scheint mit 50 immer gut zu klappen
forceHigherOrder = False

data_path = 'C:/Users/Admin/Desktop/Python/Datasets/'
'''
try:
    #raise Exception('Ich will den Datensatz neu erstellen')
    x_train = np.load(data_path+'rotatingMNIST_train.npy')
    x_test = np.load(data_path+'rotatingMNIST_test.npy')
    x_train = x_train[0:train_size]
    x_test = x_test[0:test_size]
except:
    print('Dataset is being generated. This may take a few minutes.')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[0:train_size]
    x_test = x_test[0:test_size]
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
        np.save(data_path+'rotatingMNIST_train', x_train)
        np.save(data_path+'rotatingMNIST_test', x_test)
    except:
        print('could not save Dataset')
    print('Dataset generated')


# Dim: train_size x frames x pictureWidth x pictureHeight x pictureColors
x_train = np.transpose(np.array([x_train]), (1, 4, 2, 3, 0))
print('train-shape:', x_train.shape)
'''

x_train = np.load(data_path+'SDE_Ball_train.npy')
x_test = np.load(data_path+'SDE_Ball_test.npy')
x_train = x_train[0:train_size]
x_test = x_test[0:test_size]


print('train-shape:', x_train.shape)

# Dim: train_size x frames x pictureWidth x pictureHeight x pictureColors
x_train = np.transpose(np.array([x_train]), (1, 2, 3, 4, 0))
print('train-shape:', x_train.shape)


encoder = tf.keras.models.load_model(data_path+'SDE_Zeug_Neu/encoder,Ball,M=2,e=1,l=10')
decoder = tf.keras.models.load_model(data_path+'SDE_Zeug_Neu/decoder,Ball,M=2,e=1,l=10')
ms_Net = tf.keras.models.load_model(data_path+'SDE_Zeug_Neu/ms_Net,Ball,M=2,e=1,l=10')


class SDE_Variational_Autoencoder(tf.keras.Model):
    def __init__(self, encoder, reconstructor, decoder, custom_loss, apply_ms_Net=True):
        super(SDE_Variational_Autoencoder, self).__init__()
        #self.outputs = ['a','b','c','d','e']
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor
        self.apply_ms_Net = apply_ms_Net
        self.custom_loss = custom_loss

    def fullcall(self, inputs):
        # inputs hat dim: batch_size x frames x pictureWidth x pictureHeight x pictureColors
        #print('fullcall on:',inputs, inputs.shape[1])
        frames_List = tf.split(inputs, inputs.shape[1], axis=1)
        Z_enc_List = []
        Z_enc_mean_List = []
        Z_enc_log_var_List = []
        for x in frames_List:
            # x hat dim: batch_size x 1 x pictureWidth x pictureHeight x pictureColors
            Z_enc_mean, Z_enc_log_var, Z_enc = self.encoder(x[:, 0, :, :, :])
            Z_enc_List.append(tf.transpose([Z_enc], [1, 0, 2]))
            Z_enc_mean_List.append(tf.transpose([Z_enc_mean], [1, 0, 2]))
            Z_enc_log_var_List.append(tf.transpose([Z_enc_log_var], [1, 0, 2]))
        Z_enc_List = tf.keras.layers.Concatenate(axis=1)(Z_enc_List)
        # Z_enc_List hat dim: batch_size x frames x latent_dim
        Z_enc_mean_List = tf.keras.layers.Concatenate(axis=1)(Z_enc_mean_List)
        # Z_enc_mean_List hat dim: batch_size x frames x latent_dim
        Z_enc_log_var_List = tf.keras.layers.Concatenate(axis=1)(Z_enc_log_var_List)
        # Z_enc_log_var_List hat dim: batch_size x frames x latent_dim

        Z_derivatives = derivatives(Z_enc_List)
        # Z_derivatives hat dim: batch_size x frames-M+1 x M x latent_dim
        Z_derivatives_0 = Z_derivatives[:, 0, :, :]
        # Z_derivatives_0 hat dim: batch_size x M x latent_dim

        if self.apply_ms_Net:
            Z_rec_List = self.reconstructor(Z_derivatives_0)[:, :, 0, :]
        else:
            Z_rec_List = Z_enc_List[:, :frames-M+1, :]
        # Z_rec_List hat dim: batch_size x frames-M+1 x latent_dim

        X_rec_List = []
        for i in range(frames-M+1):
            X_rec = self.decoder(Z_rec_List[:, i, :])
            X_rec = tf.transpose([X_rec], [1, 0, 2, 3, 4])
            X_rec_List.append(X_rec)
        #X_rec_List = tf.stack(X_rec_List, axis=1, name = 'output_5')
        X_rec_List = tf.keras.layers.Concatenate(axis=1)(X_rec_List)
        # X_rec_List hat dim: batch_size x frames-M+1 x pictureWidth x pictureHeight x pictureColors
        return [Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List]

    def call(self, inputs):
        Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List = self.fullcall(
            inputs)
        #CustomLoss(inputs, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List)
        return self.custom_loss(inputs, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List)


reconstructor = SDE_Tools.make_Tensorwise_Reconstructor(
    latent_dim*pictureColors, latent_dim*pictureColors, n, SDE_Time, frames-M+1, ms_Net, batch_size, applyBM=False)

SDE_VAE = SDE_Variational_Autoencoder(encoder, reconstructor, decoder, StartingLoss)
SDE_VAE.compile(optimizer='adam', loss=lambda x, arg: arg)

Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List = SDE_VAE.fullcall(
    x_test)
