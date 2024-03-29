#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from math import sin, cos, sqrt, pi, floor, ceil, exp, log
from keras import backend as K

# tf.random.set_seed(1)

# X_org hat dim: None x frames x latent_dim
# Output hat dim: None x frames-M+1 x M x latent_dim


########################################################
#Funktion um anhand von Pfaden die empirischen Ableitungen zu berechnen
#Hier berechnung als Mittelwert über 2 Stellen
def make_tensorwise_derivatives(M, frames, fps):
    def derivatives(x_org):
        length = x_org.shape[1]
        dList = [x_org]
        for i in range(1, M):
            x_last = dList[-1]
            x_derived = x_last[:, 1:, :]-x_last[:, :-1, :]
            x_derived = x_derived*fps
            dList.append(x_derived)
        dList = list(map(lambda arg: arg[:, :length-M+1, :], dList))
        return tf.stack(dList, axis=2)
    return derivatives

########################################################
#Funktion um anhand von Pfaden die empirischen Ableitungen zu berechnen
#Hier berechnung als Mittelwert über N+1 Stellen
def make_tensorwise_average_derivatives(M, N, frames, fps):
    def derivatives(x_org):
        length = x_org.shape[1]
        dList = [x_org]
        for i in range(1, M):
            x_last = dList[-1]
            average = 0
            for j in range(1, N+1):
                x_derived = x_last[:, j:, :]-x_last[:, :-j, :]
                x_derived = x_derived[:, :x_last.shape[1]-N, :]*fps
                average += x_derived/j
            dList.append(average/N)
        dList = list(map(lambda arg: arg[:, :length-(M-1)*N, :], dList))
        return tf.stack(dList, axis=2)
    return derivatives

########################################################
#Funktion um Pfade von Ito-Diffusionen zu erhalten
# Ausgabe-Dim: frames x d [=latent_dim]
def ItoDiffusion(d, n, T, frames, simulated_frames, X_0, mu, sigma):
    fps = simulated_frames/T
    E = sqrt(3/fps)
    Y = np.array(list(map(lambda i: np.random.uniform(-E, E, n), range(simulated_frames))))
    X_sim = [X_0]
    for i in range(simulated_frames):
        X_sim.append(X_sim[-1] + mu(X_sim[-1])/fps + np.matmul(sigma(X_sim[-1]), Y[i]))
    X = []
    for i in range(frames):
        X.append(X_sim[floor(i*simulated_frames/frames)])
    return X


########################################################
#Rekonstructor als Klasse definieren damit man im nachhinein noch ms_Net austauschen kann
#Funktion definieren um anhand von startwerten und den gelernten mu & sigma den Pfad der Lösung der SDE zu erhalten
class Tensorwise_Reconstructor():
    def __init__(self, latent_dim, n, T, frames, ms_Net, D_t, simulated_frames=False, applyBM=False):
        super(Tensorwise_Reconstructor, self).__init__()
        self.latent_dim = latent_dim
        self.n = n
        self.T = T
        self.frames = frames
        self.ms_Net = ms_Net
        self.D_t = D_t

        if not simulated_frames:
            simulated_frames = frames
        fps = simulated_frames/T
        self.simulated_frames = simulated_frames
        self.applyBM = applyBM

    def __call__(self, X_0_List):
        # X_0_List hat dim: None x M x latent_dim
        X_sim = [X_0_List]
        for i in range(self.simulated_frames):
            mu_sig = self.ms_Net(tf.transpose([X_sim[-1]], [1, 0, 2, 3]))[:, 0, :, :, :]
            mu = mu_sig[:,:,:,0]
            # mu hat dim: None x M x latent_dim
            if self.applyBM:
                sig = mu_sig[:,:,:,0:]

                #ACHTUNG Mogeln zum testen:
                randomPart = 0.2*tf.constant(np.random.normal(0,0.15,size=sig.shape),dtype=tf.float32)

                #Y = tf.constant(np.random.normal(0,1,size=sig.shape),dtype=tf.float32)
                #randomPart = tf.keras.layers.Multiply()([sig, Y])

                randomPart = K.sum(randomPart, axis=[3])
                nextValue = X_sim[-1] + mu*self.D_t + sqrt(self.D_t)*randomPart
            else:
                nextValue = X_sim[-1] + mu*self.D_t
            X_sim.append(nextValue)
        X = []
        for i in range(self.frames):
            X.append(X_sim[floor(i*self.simulated_frames/self.frames)])
        X = tf.stack(X, axis=1)
        # X hat dim: None x frames x M x latent_dim
        return X


########################################################
# Netzwerk zum lernen der SDE
class mu_sig_Net(tf.keras.Model):
    def __init__(self, M, latent_dim, n, akt_fun, complexity, forceHigherOrder=False):
        super(mu_sig_Net, self).__init__()
        self.latent_dim = latent_dim
        self.M = M
        self.forceHigherOrder = forceHigherOrder
        d = M*latent_dim
        #self.inp = z = tf.keras.layers.Input(shape=(latent_dim))
        self.mu_layerList = []
        self.sig_layerList = []

        # Netzwerk um mu zu lernen
        if forceHigherOrder:
            self.mu_layerList.append(tf.keras.layers.Flatten())
            self.mu_layerList.append(tf.keras.layers.Dense(complexity, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Dense(complexity, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Dense(latent_dim, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Reshape((1, latent_dim)))
        else:
            self.mu_layerList.append(tf.keras.layers.Flatten())
            self.mu_layerList.append(tf.keras.layers.Dense(complexity*M, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Dense(complexity*M, activation=akt_fun))
            #self.mu_layerList.append(tf.keras.layers.Dense(complexity*M, activation=akt_fun)) #zuletzt ohne
            self.mu_layerList.append(tf.keras.layers.Dense(M*latent_dim, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Reshape((M, latent_dim)))

        # Netzwerk um sigma zu lernen
        self.sig_layerList.append(tf.keras.layers.Flatten())
        self.sig_layerList.append(tf.keras.layers.Dense(complexity*M*n, activation=akt_fun))
        self.sig_layerList.append(tf.keras.layers.Dense(complexity*M*n, activation=akt_fun))
        #self.sig_layerList.append(tf.keras.layers.Dense(complexity*M*n, activation=akt_fun)) #zuletzt ohne
        self.sig_layerList.append(tf.keras.layers.Dense(M*latent_dim*n, activation=akt_fun))
        self.sig_layerList.append(tf.keras.layers.Reshape((M, latent_dim, n)))

        # Netzwerke zusammenfügen
        #self.merge_layer = tf.keras.layers.Concatenate(axis=-1)([mu,sig])

        #super(mu_sig_Net, self).__init__(self.inp, mu_sig, name="mu_sig_Net")

    def call(self, inputs):
        # inputs hat dim: batch_size x frames-M+1 x M x latent_dim
        frames_List = tf.split(inputs, inputs.shape[1], axis=1)
        ms_List = []
        for x in frames_List:
            # x hat dim: batch_size x 1 x M x latent_dim

            mu = x[:, 0, :, :]
            for l in self.mu_layerList:
                mu = l(mu)
            if self.forceHigherOrder:
                mu = tf.keras.layers.Concatenate(axis=1)([x[:, 0, 1:, :], mu])
            # mu hat dim: batch_size x M x latent_dim

            # eine dim an mu anhängen, um später mit sig concat machen zu können
            mu = tf.transpose([mu], (1, 2, 3, 0))
            # mu hat dim: batch_size x M x latent_dim x 1

            sig = x[:, 0, :, :]
            for l in self.sig_layerList:
                sig = l(sig)
            # sig hat dim: batch_size x M x latent_dim x n

            mu_sig = tf.keras.layers.Concatenate(axis=-1)([mu, sig])
            # mu_sig hat dim: batch_size x M x latent_dim x 1+n
            ms_List.append(mu_sig)

        ms_List = tf.stack(ms_List, axis=1)

        # ms_List hat dim: batch_size x frames-M+1 x M x latent_dim x 1+n
        return ms_List


########################################################
# Verlustfunktionen zum lernen der SDE

MAE = tf.keras.losses.MeanAbsoluteError()
MSE = tf.keras.losses.MeanSquaredError()

def make_reconstruction_Loss(M, n, Time, frames, batch_size, T_reconstructor, derivatives, norm=abs):
    def reconstruction_loss(Z_derivatives, Z_rec):
        if Z_rec is None:
            Z_0 = Z_derivatives[:, 0, :, :]
            Z_rec = T_reconstructor(Z_0)
            Z_rec = Z_rec[:, :Z_derivatives.shape[1], 0, :]
        Z_rec = Z_rec[:, :Z_derivatives.shape[1], :]
        Z_org = Z_derivatives[:, :, 0, :]
        return MAE(Z_org[:, :frames-M+1, :], Z_rec)
    return reconstruction_loss


def make_pointwise_Loss(M, latent_dim, Time, frames, ms_Net, D_t, norm=abs):
    def pointwise_loss(Z_org, ms_rec):
        # Z_org hat dim: batch_size x frames-M+1 x M x latent_dim

        mu_approx = (Z_org[:, 1:, :, :] - Z_org[:, :-1, :, :])/D_t
        # mu_approx hat dim: batch_size x frames-M x M x latent_dim

        if ms_rec is None:
            ms_rec = ms_Net(Z_org)
        # ms_rec hat dim: batch_size x frames-M+1 x M x latent_dim x 1+n

        mu = ms_rec[:, :-1, :, :, 0]
        # mu hat dim: batch_size x frames-M x M x latent_dim
        return MAE(mu, mu_approx)
    return pointwise_loss


def make_covariance_Loss(latent_dim, Time, frames, batch_size, ms_Net, D_t, norm=abs):
    def covariance_loss(Z_org, ms_rec):
        # Z_org hat dim: batch_size x frames-M+1 x M x latent_dim

        Z_delta = Z_org[:, 1:, :, :]-Z_org[:, :-1, :, :]
        # Z_org hat dim: batch_size x frames-M x M x latent_dim

        if ms_rec is None:
            ms_rec = ms_Net(Z_org)
        # ms_rec hat dim: batch_size x frames-M+1 x M x latent_dim x 1+n

        mu = ms_rec[:, :, :, :, 0]
        # mu hat dim: batch_size x frames-M+1 x M x latent_dim

        sig = ms_rec[:, :, :, :, 1:]
        # sig hat dim: batch_size x frames-M+1 x M x latent_dim x n

        rec_delta = mu*D_t
        random_delta = [Z_delta[:, :, 0, :] - rec_delta[:, :-1, 0, :]]
        random_delta = tf.transpose(random_delta, [1, 2, 3, 0])
        # random_delta hat dim: batch_size x frames[lokal] x latent_dim x 1

        covar = list(map(lambda i: tf.keras.layers.Dot(axes=2)(
            [random_delta[:, i, :, :], random_delta[:, i, :, :]]), range(random_delta.shape[1])))
        covar = tf.stack(covar, axis=1)
        covar = K.sum(covar, axis=1)
        # covar hat dim: batch_size x frames[lokal] x latent_dim x latent_dim

        sig = sig[:, :, 0, :, :]
        # sig hat dim: batch_size x frames[lokal] x latent_dim x n

        sum_sig = list(map(lambda i: tf.keras.layers.Dot(axes=2)(
            [sig[:, i, :, :], sig[:, i, :, :]]), range(sig.shape[1])))
        sum_sig = tf.stack(sum_sig, axis=1)
        theoretical_covar = K.sum(sum_sig, axis=1)*D_t
        # theoretical_covar hat dim: batch_size x frames[lokal] x latent_dim x latent_dim

        return MAE(covar, theoretical_covar) #zuletzt
        #return MSE(covar, theoretical_covar)

    return covariance_loss


#nur zum testen, gibt Überblick wie groß sigma ist.
def make_sigma_size_Loss(latent_dim, ms_Net, norm=abs):
    def sigma_size_loss(Z_org, ms_rec):
        # Z_org hat dim: batch_size x frames-M+1 x M x latent_dim

        if ms_rec is None:
            ms_rec = ms_Net(Z_org)
        # ms_rec hat dim: batch_size x frames-M+1 x M x latent_dim x 1+n

        sig = ms_rec[:, :, 0, :, 1:]
        sig_size = tf.map_fn(norm, sig)
        sig_size = K.mean(sig_size)/latent_dim
        return sig_size

    return sigma_size_loss
