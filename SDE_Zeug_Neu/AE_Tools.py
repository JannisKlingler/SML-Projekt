import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy as sp
from keras import backend as K
import SDE_Tools
tfd = tfp.distributions
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


class LocalEncoder(tf.keras.Model):
    def __init__(self, latent_dim, M, pictureWidth, pictureHeight, pictureColors, act, complexity=1, variational=False):
        super(LocalEncoder, self).__init__()
        self.variational = variational
        self.d = latent_dim
        self.inp = z = tf.keras.layers.Input(shape=(pictureWidth, pictureHeight, M*pictureColors))
        forwardList = []
        w,h = pictureWidth, pictureHeight
        i=0
        while (2*w*w > pictureWidth and 2*h*h > pictureHeight):
            r_w, r_h = w%2, h%2
            forwardList.append(tf.keras.layers.Conv2DTranspose((2**i)*complexity*pictureColors, (3+r_w, 3+r_h), activation=act))
            forwardList.append(tf.keras.layers.MaxPooling2D((2, 2)))
            w = w//2 - 1
            h = h//2 - 1
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
        a = z
        for l in self.layerList:
            a = l(a)
            #print('applied layer:',a)
        if self.variational:
            v_mu = a[:,0,:]
            v_log_sig = a[:,1,:]
            v_sig = K.exp(v_log_sig)
            v = K.random_normal(shape=[len(v_mu),self.d])
            v = K.map_fn(lambda x: x[0]*x[1], [v, v_sig], dtype=tf.float32)
            v = K.map_fn(lambda x: x[0]+x[1], [v, v_mu], dtype=tf.float32)
            a = tf.stack([v_mu, v_log_sig, v], axis=1)
        #print('Encoder built')
        return a

class FramewiseDecoder(tf.keras.Model):
    def __init__(self, latent_dim, pictureWidth, pictureHeight, pictureColors, act, complexity=1):
        super(FramewiseDecoder, self).__init__()
        self.d = latent_dim
        self.inp = z = tf.keras.layers.Input(shape=(self.d))
        backwardList = []
        w,h = pictureWidth, pictureHeight
        i=0
        while (2*w*w > pictureWidth and 2*h*h > pictureHeight):
            r_w, r_h = w%2, h%2
            backwardList.append(tf.keras.layers.Conv2DTranspose((2**i)*complexity*pictureColors, (3+r_w, 3+r_h), activation=act))
            backwardList.append(tf.keras.layers.UpSampling2D(size=(2, 2)))
            w = w//2 - 1
            h = h//2 - 1
            i += 1
        backwardList.reverse()

        self.layerList = []
        self.layerList.append(tf.keras.layers.Dense(w * h *(2**i)*complexity*pictureColors, activation=act))
        self.layerList.append(tf.keras.layers.Reshape((w, h, (2**i)*complexity*pictureColors)))
        self.layerList += backwardList
        self.layerList.append(tf.keras.layers.Conv2D(pictureColors, (3, 3), padding="same",activation=act))


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
            print('VAE enc:',enc.shape)
            mean, variance, v = tf.split(enc, 3, axis=1)
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

#Hiermit stimmt etwas nicht
def make_VAE_loss(encoder, frames):
    def VAE_loss(X_org, X_rec):
        X_encInp = tf.transpose([X_org],(1,2,3,0))
        print('X_org:',X_org)
        print('X_rec:',X_rec)
        v_mu, v_log_sig, _ = tf.split(encoder(X_encInp), 3, axis=1)
        print('mu in loss:',v_mu)
        r1 = 784. * frames * K.mean(tf.keras.losses.binary_crossentropy(X_org, X_rec))
        kl_div = .5 * K.sum(1. + 2. * v_log_sig - K.square(v_mu) - 2. * K.exp(v_log_sig))
        return r1 - kl_div
    return VAE_loss




def make_binary_crossentropy_rec_loss(M):
    b = M//2+1
    a = M-b
    def reconstruction_loss(X_org, X_rec):
        print('GGGGG0 X_org:',X_org.shape)
        X_values = X_org[:,:,:,:,a]
        print('GGGGG X_org:',X_values.shape)
        X_rec = X_rec[:,:,:,:,0]
        print('X_rec:',X_rec.shape)
        #if len(X_values[0]) > len(X_rec[0]):
            #print('cutting X_org to {} frames'.format(len(X_rec[0])))
            #X_values = X_values[:,:len(X_rec[0]),:,:]
        r = tf.keras.losses.binary_crossentropy(X_values, X_rec)
        return r
    return reconstruction_loss
