#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from math import sin, cos, sqrt, pi, floor, ceil, exp, log
from keras import backend as K


########################################################
# Definition der SDE_VAE

class SDE_Variational_Autoencoder(tf.keras.Model):
    def __init__(self, M, N, encoder, derivatives, reconstructor, decoder, custom_loss, apply_reconstructor=True):
        super(SDE_Variational_Autoencoder, self).__init__()
        #self.outputs = ['a','b','c','d','e']
        self.M = M
        self.N = N
        self.encoder = encoder
        self.decoder = decoder
        self.derivatives = derivatives
        self.reconstructor = reconstructor
        self.apply_reconstructor = apply_reconstructor
        self.custom_loss = custom_loss

    def fullcall(self, inputs):
        # inputs hat dim: batch_size x frames x pictureWidth x pictureHeight x pictureColors
        #print('fullcall on:',inputs, inputs.shape[1])
        frames = inputs.shape[1]
        frames_List = tf.split(inputs, frames, axis=1)
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

        Z_derivatives = self.derivatives(Z_enc_List)
        print('Z_derivatives:',Z_derivatives.shape)
        # Z_derivatives hat dim: batch_size x frames-(M-1)*N x M x latent_dim
        Z_derivatives_0 = Z_derivatives[:, 0, :, :]
        # Z_derivatives_0 hat dim: batch_size x M x latent_dim

        if self.apply_reconstructor:
            Z_rec_List = self.reconstructor(Z_derivatives_0)[:, :, 0, :]
        else:
            Z_rec_List = Z_enc_List[:, :frames-(self.M-1)*self.N, :]
        # Z_rec_List hat dim: batch_size x frames-(M-1)*N x latent_dim

        X_rec_List = []
        for i in range(frames-(self.M-1)*self.N):
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
