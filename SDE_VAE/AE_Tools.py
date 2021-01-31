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
# Encoder definieren

#Encoder, der besonders gut mit dem MNIST-Datensatz funktioniert
def make_MNIST_encoder(latent_dim):
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



#Encoder, der mehrere Frames neben einander betrachtet
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
        if self.variational:
            self.layerList.append(tf.keras.layers.Dense(2*self.d, activation=act))
            self.layerList.append(tf.keras.layers.Reshape((2,self.d)))
        else:
            self.layerList.append(tf.keras.layers.Dense(self.d, activation=act))

    def call(self, z):
        a = z
        for l in self.layerList:
            a = l(a)
        if self.variational:
            v_mu = a[:,0,:]
            v_log_sig = a[:,1,:]
            v_sig = K.exp(v_log_sig)
            v = K.random_normal(shape=[len(v_mu),self.d])
            v = tf.map_fn(lambda x: x[0]*x[1], [v, v_sig], dtype=tf.float32)
            v = tf.map_fn(lambda x: x[0]+x[1], [v, v_mu], dtype=tf.float32)
            a = tf.stack([v_mu, v_log_sig, v], axis=1)
        return a

#Encoder, der nur auf ein einzelnes Frame angewendet werden kann
#eigentlich überflüsst, weil Local-Encoder das auch kann
class FramewiseEncoder(tf.keras.Model):
    def __init__(self, latent_dim, pictureWidth, pictureHeight, pictureColors, act, complexity=1, variational=False):
        super(FramewiseEncoder, self).__init__()
        self.variational = variational
        self.d = latent_dim
        self.inp = z = tf.keras.layers.Input(shape=(pictureWidth, pictureHeight, pictureColors))
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

        if self.variational:
            self.layerList.append(tf.keras.layers.Dense(2*self.d, activation=act))
            self.layerList.append(tf.keras.layers.Reshape((2,self.d)))
        else:
            self.layerList.append(tf.keras.layers.Dense(self.d, activation=act))

    def call(self, z):
        a = z
        for l in self.layerList:
            a = l(a)
        if self.variational:
            v_mu = a[:,0,:]
            v_log_sig = a[:,1,:]
            v_sig = K.exp(v_log_sig)
            v = K.random_normal(shape=[len(v_mu),self.d])
            v = tf.map_fn(lambda x: x[0]*x[1], [v, v_sig], dtype=tf.float32)
            v = tf.map_fn(lambda x: x[0]+x[1], [v, v_mu], dtype=tf.float32)
            return [v_mu, v_log_sig, v]
        return a


########################################################
# Encoder definieren

#Encoder, der besonders gut mit dem MNIST-Datensatz funktioniert
def make_MNIST_decoder(latent_dim):
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



#Decoder um einlend Frames wieder herzustellen
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
        for l in self.layerList:
            a = l(a)
        return a


########################################################
# Modelle definieren


class SimpleAutoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        frames_List = tf.split(inputs, len(inputs[0]), axis=1)
        rec_List = []
        for x in frames_List:
            rec_List.append(self.decoder(self.encoder(x[:,0,:,:,:])))
        rec_List = tf.stack(rec_List, axis=1)
        return rec_List

#nur Modell. für einen VAE braucht man auch die richtige loss-Funktion
class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        frames_List = tf.split(inputs, len(inputs[0]), axis=1)
        rec_List = []
        for x in frames_List:
            enc = self.encoder(x[:,0,:,:,:])
            mean, variance, v = enc
            rec_List.append(self.decoder(v))
        rec_List = tf.stack(rec_List, axis=1)
        return rec_List


########################################################
# Loss-Funktionen definieren

MAE = tf.keras.losses.MeanAbsoluteError()

def make_simple_reconstruction_loss(batchsize):
    def reconstruction_loss(X_org, X_rec):
        return MAE(X_org, X_rec)
    return reconstruction_loss


def make_binary_crossentropy_rec_loss(frames):
    def reconstruction_loss(X_org, X_rec):
        X_values = X_org[:,:X_rec.shape[1],:,:,:]
        return tf.keras.losses.binary_crossentropy(X_values, X_rec)
    return reconstruction_loss



#
