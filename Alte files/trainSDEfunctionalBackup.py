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
nrBrMotions = 1
epochs = 3

akt_fun = 'relu'

frames = 50
simulated_frames = 100
T = 3
fps = frames//T
Ntrain = 10000
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
    s = np.array([[0.2],[0.1]])
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
def Reconstructor(d, n, T, frames, simulated_frames, X_0_List, ms_Net, NrReconstr, applyBM=True):
    fps = simulated_frames/T
    E = sqrt(3/fps)
    if applyBM:
        Y = np.random.uniform(-E,E,(NrReconstr,simulated_frames,n))
    else:
        Y = np.zeros((NrReconstr,simulated_frames,n))
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


#Input-Form: frames x d
def squareVariation(X):
    frames, d = X.shape
    return 0







########################################################
#Datensatz erstellen
#Dimensionen: Ntrain x frames x latent_dim
x_train = np.array(list(map(lambda i : ItoDiffusion(d, n, T, frames, simulated_frames, X_0[i], mu, sigma) , range(Ntrain))))
x_test = np.array(list(map(lambda i : ItoDiffusion(d, n, T, frames, simulated_frames, X_0[i], mu, sigma) , range(Ntest))))



########################################################
#Trainigsdatensatz umstellen um mu und sigma zu Lernen

#Input braucht Form: Ntrain x frames x latent_dim
#Output hat Form: 2 x (Ntrain*(frames-1)) x latent_dim
def make_SDE_training_data(x_train):
    x_train = np.array(x_train)
    Ntrain, frames, d = x_train.shape
    L = list(map(lambda i: x_train[i,1:,:]-x_train[i,0:-1,:] ,range(Ntrain)))
    x_train_delta_list = np.concatenate(L,axis=0)
    L = list(map(lambda i: x_train[i,0:-1,:],range(Ntrain)))
    x_train_value_list = np.concatenate(L,axis=0)

    return (x_train_value_list, x_train_delta_list)

x_train_value_list, x_train_delta_list = make_SDE_training_data(x_train)



########################################################
#Netzwerk zum lernen der SDE aufbauen
R = 20
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
alpha = 1
beta = 10
norm1 = abs
norm2 = lambda arg: tf.math.sqrt(abs(arg))

def norm2(arg):
    print(arg,abs(arg))
    return abs(arg)


def pointwise_loss(X_delta, mu_sig):
    b = len(X_delta) # = frames-1
    mu = mu_sig[:,:,0]
    sig = mu_sig[:,:,1:]
    fps = (b+1)/T
    '''
    E = sqrt(3/fps)
    Y = np.array(list(map(lambda i: np.random.uniform(-E,E,n), range(b))))
    Y = tf.constant(Y)
    Y = tf.keras.layers.Reshape((n, 1))(Y)
    mult_sig_Y = tf.keras.layers.Dot((2,1))([sig,Y])
    mult_sig_Y = tf.keras.layers.Reshape([d])(mult_sig_Y)
    '''
    #rec_delta = tf.keras.layers.Add()([mu/fps,mult_sig_Y])
    rec_delta = mu/fps
    Difference = tf.keras.layers.Subtract()([X_delta, rec_delta])
    Diff = tf.map_fn(norm1,Difference)
    #Diff = tf.keras.layers.Lambda(norm1)(Difference)
    #Diff = tf.keras.layers.Add()([Diff[:,0],Diff[:,1]])
    Diff = K.sum(Diff)


    sq_var = tf.keras.layers.Multiply()([Difference,Difference])
    sq_var = K.sum(sq_var,axis=0)

    sig_squared = tf.keras.layers.Dot((2,2))([sig,sig])
    sum_sig_squared = K.sum(sig_squared, axis=2)
    int_sum_sig_squared = K.sum(sum_sig_squared, axis=0)/fps

    Diff_sq_var = sq_var - int_sum_sig_squared
    #Diff_sq_var = tf.keras.layers.Lambda(norm2)(Diff_sq_var)
    Diff_sq_var = tf.map_fn(norm2,Diff_sq_var)
    Diff_sq_var = K.sum(Diff_sq_var)

    return alpha*Diff + beta*Diff_sq_var






########################################################
#Model trainieren

ms = mu_sig_Net(d)

ms.compile(optimizer='adam', loss=pointwise_loss,)
ms.fit(x_train_value_list, x_train_delta_list, epochs=epochs, batch_size=frames-1)





########################################################
#Ergebnisse plotten

fig, axs = plt.subplots(3, 5)


x = np.linspace(0,T,frames)
x_2 = np.array([x,np.zeros(frames)])
x_2 = np.transpose(x_2, [1,0])
y = ms.predict(x_2)

axs[0, 1].plot(x,y[:,:,0])
axs[0, 1].set_title('Rekonstr.-mu')

a = np.array(list(map(mu, x_2)))
axs[0, 0].plot(x,a)
axs[0, 0].set_title('Original-mu')


sig_sq = np.array(list(map(lambda j: np.matmul(y[j,:,1:], np.transpose(y[j,:,1:],(1,0))), range(len(y)))))
#print('Shape5:',sig_sq.shape)
axs[1, 3].plot(x,sig_sq[:,0,:])
axs[1, 3].set_title('Rekonstr.-sig1')
axs[1, 3].axis((0,T,0,0.05))
axs[1, 4].plot(x,sig_sq[:,1,:])
axs[1, 4].set_title('Rekonstr.-sig2')
axs[1, 4].axis((0,T,0,0.05))


a = np.array(list(map(sigma, x_2)))
#print(a[0,:,:])
sig_sq = np.array(list(map(lambda j: np.matmul(a[j,:,:], np.transpose(a[j,:,:],(1,0))), range(len(a)))))
#print(sig_sq[0,:,:])
axs[1, 0].plot(x,sig_sq[:,0,:])
axs[1, 0].set_title('Original-sig1')
axs[1, 0].axis((0,T,0,0.05))
axs[1, 1].plot(x,sig_sq[:,1,:])
axs[1, 1].set_title('Original-sig2')
axs[1, 1].axis((0,T,0,0.05))



NrRec = 1
R = Reconstructor(d, n, T, frames, simulated_frames, np.zeros((NrRec,d)), ms, NrRec, applyBM=False)
axs[2, 0].plot(x,R[0,:,:])
axs[2, 0].axis((0,T,-1,3))
axs[2, 0].set_title('SDE ohne BB (rec)')

for i in range(4):
    axs[2, i+1].plot(x,x_test[i,:,:])
    axs[2, i+1].axis((0,T,-0.5,2))
    axs[2, i+1].set_title('Ziehung der SDE')



plt.show()
