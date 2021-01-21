import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp
import scipy as sp
from keras import backend as K
import SDE_Tools
#tfd = tfp.distributions
# import datasets as data




########################################################
# Datensatz für Encoder erstellen

def make_training_data(x_train, train_size, frames, M):
    b = M//2+1
    a = M-b
    #print('a,b:',a,b)

    L = []
    for i in range(train_size):
        L1 = []
        for j in range(a,frames-b+1):
            L2 = list(map(lambda k: x_train[i,j-a+k,:,:,:],range(M)))
            L1.append(np.concatenate(L2,axis=-1))
        L.append(L1)

    #Dim: train_size x (frames-M+1) x pictureWidth x pictureHeight x (M*pictureColors)
    x_train_longlist = np.stack(L,axis=0)
    return x_train_longlist



########################################################
# Encoder definieren

def make_Clemens_encoder(latent_dim):
    encoder_input = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(encoder_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    μ = tf.keras.layers.Dense(latent_dim)(x)
    log_σ = tf.keras.layers.Dense(latent_dim)(x)

    z = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
        shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([μ, log_σ])

    Clemens_encoder = tf.keras.Model(encoder_input, [μ, log_σ, z])
    return Clemens_encoder



class LocalEncoder(tf.keras.Model):
    def __init__(self, latent_dim, M, pictureWidth, pictureHeight, pictureColors, act, complexity=1, variational=False):
        super(LocalEncoder, self).__init__()
        self.variational = variational
        self.d = latent_dim
        self.inp = z = tf.keras.layers.Input(shape=(pictureWidth, pictureHeight, M*pictureColors))
        forwardList = []
        w,h = pictureWidth, pictureHeight
        i=0
        while (w*w > pictureWidth and h*h > pictureHeight):
            r_w, r_h = w%2, h%2
            forwardList.append(tf.keras.layers.Conv2D((2**i)*complexity*pictureColors, (3+r_w, 3+r_h), activation=act))
            forwardList.append(tf.keras.layers.MaxPooling2D((2, 2)))
            w = (w-r_w)//2 -1
            h = (h-r_h)//2 -1
            i += 1

        self.layerList = [tf.keras.layers.Conv2D(complexity*pictureColors, (3, 3), padding="same",activation=act)]
        self.layerList += forwardList
        self.layerList.append(tf.keras.layers.Flatten())
        self.layerList.append(tf.keras.layers.Dense((2**i)*complexity*pictureColors, activation=act))
        #self.layerList.append(tf.keras.layers.Dense(complexity*self.d, activation=act))
        if self.variational:
            self.layerList.append(tf.keras.layers.Dense(2*self.d, activation=act))
            self.layerList.append(tf.keras.layers.Reshape((2,self.d)))
        else:
            self.layerList.append(tf.keras.layers.Dense(self.d, activation=act))

    def call(self, z):
        print('Ecoding:',z.shape)
        a = z
        for l in self.layerList:
            a = l(a)
            #print('applied layer:',a)
        if self.variational:
            v_mu = a[:,0,:]
            v_log_sig = a[:,1,:]
            v_sig = K.exp(v_log_sig)
            v = K.random_normal(shape=[len(v_mu),self.d])
            v = tf.map_fn(lambda x: x[0]*x[1], [v, v_sig], dtype=tf.float32)
            v = tf.map_fn(lambda x: x[0]+x[1], [v, v_mu], dtype=tf.float32)
            a = tf.stack([v_mu, v_log_sig, v], axis=1)
        #print('Encoder built')
        return a

def make_Clemens_decoder(latent_dim):
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(decoder_input)
    x = tf.keras.layers.Dense(4 * 4 * 64, activation='relu')(x)
    x = tf.keras.layers.Reshape((4, 4, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(16, (3, 3), activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    decoder_output = tf.keras.layers.Conv2DTranspose(1, (4, 4), padding='same', activation='sigmoid')(x)

    Clemens_decoder = tf.keras.Model(decoder_input, decoder_output)
    return Clemens_decoder



class FramewiseDecoder(tf.keras.Model):
    def __init__(self, latent_dim, pictureWidth, pictureHeight, pictureColors, act, complexity=1):
        super(FramewiseDecoder, self).__init__()
        self.d = latent_dim
        self.inp = z = tf.keras.layers.Input(shape=(self.d))
        backwardList = []
        w,h = pictureWidth, pictureHeight
        i=0
        while (w*w > pictureWidth and h*h > pictureHeight):
            r_w, r_h = w%2, h%2
            backwardList.append(tf.keras.layers.Conv2DTranspose((2**(i-1))*complexity*pictureColors, (3+r_w, 3+r_h), activation=act))
            backwardList.append(tf.keras.layers.UpSampling2D(size=(2, 2)))
            w = (w-r_w)//2 - 1
            h = (h-r_h)//2 - 1
            i += 1
        backwardList.reverse()

        self.layerList = []
        self.layerList.append(tf.keras.layers.Dense((2**(i))*complexity*pictureColors, activation=act))
        self.layerList.append(tf.keras.layers.Dense(w * h *(2**(i-1))*complexity*pictureColors, activation=act))
        self.layerList.append(tf.keras.layers.Reshape((w, h, (2**(i-1))*complexity*pictureColors)))
        self.layerList += backwardList
        self.layerList.append(tf.keras.layers.Conv2DTranspose(pictureColors, (3, 3), padding="same",activation='sigmoid'))


    def call(self, z):
        a = z
        #print('Decoding input:',z)
        for l in self.layerList:
            a = l(a)
            #print('applied layer:',a)
        #print('Decoder built')
        return a


class SimpleAutoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        print('AE_Net input:',inputs.shape)
        frames_List = tf.split(inputs, len(inputs[0]), axis=1)
        rec_List = []
        for x in frames_List:
            rec_List.append(self.decoder(self.encoder(x[:,0,:,:,:])))
        rec_List = tf.stack(rec_List, axis=1)
        print('AE_Net output:',rec_List.shape)
        return rec_List

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        print('VAE_Net input:',inputs.shape)
        frames_List = tf.split(inputs, len(inputs[0]), axis=1)
        rec_List = []
        for x in frames_List:
            enc = self.encoder(x[:,0,:,:,:])
            print('VAE enc:',enc[-1].shape)
            mean, variance, v = enc
            rec_List.append(self.decoder(v))
        rec_List = tf.stack(rec_List, axis=1)
        print('VAE_Net output:',rec_List.shape)
        #out = self.encoder(z)
        #v_mu, v_log_sig, v = tf.split(out, 3, axis=1)
        return rec_List


######################################
#Hiermit stimmt etwas nicht
def make_simple_reconstruction_loss(batchsize):
    def reconstruction_loss(X_org, X_rec):
        #X_org = tf.split(X_org, batchSize, axis=0)
        #X_org = tf.stack(X_org, axis=0)
        print('X_org:',X_org)
        print('X_rec:',X_rec)
        Diff = X_org - X_rec
        Diff = tf.map_fn(abs, Diff)
        Diff = K.mean(Diff)
        return Diff
    return reconstruction_loss


def make_VAE_loss(encoder, decoder, frames):
    def VAE_loss(X_org, X_rec):
        X_org = X_org[:,0,:,:,:]
        print('X_org0:',X_org.shape)
        #X_encInp = tf.transpose([X_org],(1,2,3,0))
        #print('X_org:',X_org.shape)
        v_mu, v_log_sig, z = encoder(X_org)
        X_rec = decoder(z)
        print('X_rec:',X_rec.shape)
        print('mu in loss:',v_mu.shape)
        log_p_xz = 784. * frames * K.mean(tf.keras.losses.binary_crossentropy(X_org, X_rec))
        kl_div = .5 * K.sum(1. + 2. * v_log_sig - K.square(v_mu) - 2. * K.exp(v_log_sig))
        return log_p_xz #- kl_div
    return VAE_loss




def make_binary_crossentropy_rec_loss(M,frames):
    b = M//2+1
    a = M-b
    def reconstruction_loss(X_org, X_rec):
        print('GGGGG0 X_org:',X_org.shape)
        X_values = X_org[:,:frames-M+1,:,:,:]
        print('GGGGG X_org:',X_values.shape)
        #X_rec = X_rec[:,:,:,:,:]
        #print('len:',len(X_rec))
        print('X_rec:',X_rec.shape)
        #if len(X_values[0]) > len(X_rec[0]):
            #print('cutting X_org to {} frames'.format(len(X_rec[0])))
            #X_values = X_values[:,:len(X_rec[0]),:,:]
        r = tf.keras.losses.binary_crossentropy(X_values, X_rec)
        return r
    return reconstruction_loss



'''

########################################################
# %% hyperparameter
epochs = 1
latent_dim = 5  # Dimensionality for latent variables. 20-30 works fine.
batch_size = 100  # ≥100 as suggested by Kingma in Autoencoding Variational Bayes.
train_size = 60000  # Data points in train set. Choose accordingly to dataset size.
test_size = 100  # Data points in test set. Choose accordingly to dataset size.
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

print('train starting shape:', x_train.shape)

# Dim: train_size x frames x pictureWidth x pictureHeight x pictureColors
x_train = np.transpose(np.array([[x_train]]), (2, 0, 3, 4, 1))
print('train-shape:', x_train.shape)


# Dim: test_size x frames x pictureWidth x pictureHeight x pictureColors
x_test = np.transpose(np.array([[x_test]]), (2, 0, 3, 4, 1))


########################################################
# Datensatz für Encoder erstellen

# Dim: train_size*(frames-M+1) x pictureWidth x pictureHeight x (M*pictureColors)
x_train_longlist = AE_Tools.make_training_data(x_train, train_size, frames, M)
print('new_train_shape:', x_train_longlist.shape)


######################################

# P_dec = AE_Tools.FramewiseDecoder(latent_dim, pictureWidth,
# pictureHeight, pictureColors, act, complexity=complexity)
decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(128, activation='relu')(decoder_input)
x = tf.keras.layers.Dense(4 * 4 * 64, activation='relu')(x)
x = tf.keras.layers.Reshape((4, 4, 64))(x)
x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu')(x)
x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
x = tf.keras.layers.Conv2DTranspose(16, (3, 3), activation='relu')(x)
x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
decoder_output = tf.keras.layers.Conv2DTranspose(1, (4, 4), padding='same', activation='sigmoid')(x)


decoder = tf.keras.Model(decoder_input, decoder_output)
VAE = tf.keras.Model(encoder_input, decoder(encoder(encoder_input)[-1]))


# P_enc = AE_Tools.LocalEncoder(latent_dim, M, pictureWidth, pictureHeight,
# pictureColors, act, complexity=complexity, variational=True)
# AE = AE_Tools.SimpleAutoencoder(P_enc, P_dec)
# VAE = AE_Tools.VariationalAutoencoder(P_enc, decoder)

# ms_Net = SDE_Tools.mu_sig_Net(latent_dim, n, act, 10)
# reconstructor = SDE_Tools.make_Tensorwise_Reconstructor(latent_dim*pictureColors, n, T, frames, ms_Net, batch_size)

loss = AE_Tools.make_binary_crossentropy_rec_loss(M)

# AE.compile(optimizer='adam', loss=loss)
# AE.fit(x_train_longlist, x_train_longlist[:,:,:,:,:], epochs=epochs, batch_size=batch_size, shuffle=False)


VAE.compile(optimizer='adam', loss=loss)
VAE.fit(x_train_longlist[:, 0, :, :, :], x_train_longlist[:, 0, :, :, :],
        epochs=epochs, batch_size=batch_size, shuffle=False)
'''

#
