import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import scipy as sp
import models
import lossfunctions
from random import choice, uniform
from math import sin, cos, sqrt, pi, floor, ceil, exp, log
from keras import backend as K

tf.random.set_seed(1)



########################################################
#Parameter festlegen

latent_dim = 2
nrBrMotions = 3
epochs = 5

akt_fun = 'relu'

frames = 10
simulated_frames = 10
T = 3
fps = frames//T
Ntrain = 1000
Ntest = 10
d = latent_dim
n = nrBrMotions



########################################################
#Definitionen

#(Ntrain x d)-Array der Startwerte
#X_0 = np.array([np.zeros(Ntrain), np.random.uniform(0,1,Ntrain)])
X_0 = np.array([np.zeros(Ntrain), np.zeros(Ntrain)])
X_0 = np.transpose(X_0, [1,0])

#mu : R^d -> R^d
def mu(x):
    m = np.array([1,x[0]])
    return m

#sigma: R^d -> R^(nxd)
def sigma(x):
    s = np.array([[0.2,0,0],[0,0.05,0.1]])
    #s = np.zeros((d,n))
    return s

#Ausgabe-Dim: frames x d [=latent_dim]
def ItoDiffusion(d, n, T, frames, simulated_frames, X_0, mu, sigma):
    fps = simulated_frames/T
    E = sqrt(3/fps)
    Y = np.array(list(map(lambda i: np.random.uniform(-E,E,n), range(simulated_frames))))
    X_sim = [X_0]
    for i in range(simulated_frames):
        X_sim.append( X_sim[-1] + mu(X_sim[-1])/fps + np.matmul(sigma(X_sim[-1]), Y[i]))
    X = []
    for i in range(frames):
        X.append(X_sim[floor(i*simulated_frames/frames)])
    return X

#rekonstruiere den Pfad nach den gelernten mu,sigma (Parallel)
#Output-Dim : NrReconstr x frames x d
#X_0 hat Form: NrReconstr x d
def Reconstructor(d, n, T, frames, simulated_frames, X_0_List, ms_Net, NrReconstr):
    fps = simulated_frames/T
    E = sqrt(3/fps)
    Y = np.random.uniform(-E,E,(NrReconstr,simulated_frames,n))
    X_sim = [X_0_List]
    for i in range(simulated_frames):
        ms = ms_Net.predict(X_sim[-1])
        mu = ms[:,:,0]
        sig = ms[:,:,1:]
        mult_sig_Y = np.array(list(map(lambda j: np.matmul(sig[j,:,:], Y[j,i,:]), range(NrReconstr))))
        X_sim.append( X_sim[-1] + mu/fps + mult_sig_Y)
    X = []
    for i in range(frames):
        X.append(X_sim[floor(i*simulated_frames/frames)])
    X = np.array(X)
    X = np.transpose(X,(1,0,2))
    return X







########################################################
#Datensatz erstellen
#Dimensionen: Ntrain x frames x latent_dim
x_train = list(map(lambda i : ItoDiffusion(d, n, T, frames, simulated_frames, X_0[i], mu, sigma) , range(Ntrain)))
x_test = list(map(lambda i : ItoDiffusion(d, n, T, frames, simulated_frames, X_0[i], mu, sigma) , range(Ntest)))



########################################################
#Trainigsdatensatz umstellen um mu und sigma zu Lernen

#Dimensionen: Ntrain x frames-1 x latent_dim
x_train_delta = list(map(lambda X: list(np.array(X)[1:,:]-np.array(X)[0:-1,:]) ,x_train))

#Dimensionen: Ntrain x frames-1 x 2 x latent_dim
#x_train_value_and_delta = list(map(lambda N: list(map(lambda t: np.array([x_train[N][t,:],x_train_delta[N][t,:]]), range(frames-1))),range(Ntrain)))

#Dimensionen: (Ntrain*(frames-1)) x latent_dim
x_train_delta_list = []
for L in x_train_delta:
    x_train_delta_list += L
x_train_delta_list = np.array(x_train_delta_list)

x_train_value_list = []
for L in x_train:
    x_train_value_list += L[0:-1]
x_train_value_list = np.array(x_train_value_list)

x_train = np.array(x_train)


########################################################
#Netzwerk zum lernen der SDE aufbauen
R = 10
class mu_sig_Net(tf.keras.Model):
    def __init__(self, d):
        self.inp = z = tf.keras.layers.Input(shape=(d))

        #Netzwerk um mu zu lernen
        mu = tf.keras.layers.Dense(R*d, activation=akt_fun)(z)
        mu = tf.keras.layers.Dense(R*d, activation=akt_fun)(mu)
        mu = tf.keras.layers.Dense(d, activation=akt_fun)(mu)
        mu = tf.keras.layers.Reshape((d, 1))(mu)

        #Netzwerk um sigma zu lernen
        sig = tf.keras.layers.Dense(R*d*n, activation=akt_fun)(z)
        sig = tf.keras.layers.Dense(R*d*n, activation=akt_fun)(sig)
        sig = tf.keras.layers.Dense(d*n, activation=akt_fun)(sig)
        sig = tf.keras.layers.Reshape((d, n))(sig)

        #Netzwerke zusammenfügen
        mu_sig = tf.keras.layers.Concatenate(axis=-1)([mu,sig])

        super(mu_sig_Net, self).__init__(
            self.inp, mu_sig, name="mu_sig_Net")



########################################################
#Verlustfunktion definieren

def Pointwise_Loss(X_delta, mu_sig):
    b = len(X_delta) # = frames-1
    mu = mu_sig[:,:,0]
    sig = mu_sig[:,:,1:]
    sig = tf.keras.layers.Permute((2,1))(sig)
    fps = (b+1)/T
    E = sqrt(3/fps)
    Y = np.array(list(map(lambda i: np.random.uniform(-E,E,n), range(b))))
    Y = tf.constant(Y)
    Y = tf.keras.layers.Reshape((1, n))(Y)
    mult_sig_Y = tf.keras.layers.Dot((1,2))([sig,Y])
    mult_sig_Y = tf.keras.layers.Reshape([d])(mult_sig_Y)
    rec_delta = tf.keras.layers.Add()([mu/fps,mult_sig_Y])
    D = tf.keras.layers.Subtract()([X_delta, rec_delta])
    D = tf.keras.layers.Lambda(abs)(D)
    D = tf.keras.layers.Add()([D[:,0],D[:,1]])
    return K.sum(D)





########################################################
#Model trainieren

ms = mu_sig_Net(d)

ms.compile(optimizer='adam', loss=Pointwise_Loss,)
ms.fit(x_train_value_list, x_train_delta_list, epochs=epochs, batch_size=frames-1)













fig, axs = plt.subplots(7, 6)

x = np.linspace(0,T,frames)
x_2 = np.array([x,np.zeros(frames)])
x_2 = np.transpose(x_2, [1,0])
y = ms.predict(x_2)
for i in range(d):
    axs[0, 1].plot(x,y[:,i,0])
axs[0, 1].set_title('Rekonstr.-mu')

for i in range(d):
    a = np.array(list(map(mu, x_2)))
    axs[0, 0].plot(x,a)
axs[0, 0].set_title('Original-mu')

########################################################
#Datensatz zeichnen
NrRec = 18
R = Reconstructor(d, n, T, frames, simulated_frames, np.zeros((NrRec,d)), ms, NrRec)

for i in range(18):
    j,k = i//6 , i%6
    for l in range(d):
        axs[2*j+1, k].plot(list(np.linspace(0,T,frames)),x_train[i,:,l])
    axs[2*j+1, k].axis((0,T,-0.5,2))
    axs[2*j+2, k].plot(list(np.linspace(0,T,frames)),R[i,:,:])
    axs[2*j+2, k].axis((0,T,-1,3))
plt.show()
