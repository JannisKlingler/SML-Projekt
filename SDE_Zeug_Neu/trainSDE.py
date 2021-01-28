import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import scipy as sp
from math import sin, cos, sqrt, pi, floor, ceil, exp, log
from keras import backend as K
import SDE_Tools

tf.random.set_seed(1)


########################################################
# Parameter festlegen

latent_dim = 6
nrBrMotions = 1
epochs = 10
M = 2
forceHigherOrder = False

akt_fun = 'tanh'

frames = 20
simulated_frames = frames
T = 10
#T = 2*pi
fps = frames//T
Ntrain = 5000
Ntest = 10

d = M*latent_dim
n = nrBrMotions
batch_size = 50
complexity = 100
expected_SDE_complexity = 20


########################################################
# Definitionen

# (Ntrain x d)-Array der Startwerte
#X_0 = np.array([np.zeros(Ntrain), np.random.uniform(0,1,Ntrain)])
X_0 = np.array([np.zeros(Ntrain), np.ones(Ntrain)])
X_0 = np.transpose(X_0, [1, 0])

# mu : R^d -> R^d


def mu(x):
    m = np.array([x[1], -(2*pi/T)**2*x[0]])
    #m = np.array([x[1],1])
    return m

# sigma: R^d -> R^(nxd)


def sigma(x):
    s = np.array([[0], [1]])
    #s = np.zeros((d,n))
    return s


########################################################
# Datensatz erstellen
# Dimensionen: Ntrain x frames x latent_dim
#x_train = np.array(list(map(lambda i : SDE_Tools.ItoDiffusion(2, n, T, frames, simulated_frames, X_0[i], mu, sigma) , range(Ntrain))))
#x_test = np.array(list(map(lambda i : SDE_Tools.ItoDiffusion(2, n, T, frames, simulated_frames, X_0[i], mu, sigma) , range(Ntest))))
#x_train = x_train[:Ntrain,:,:-1]
x_train = np.load('C:/Users/bende/Documents/Uni/SML-Projekt/SDE_Zeug_Neu/SDE_Daten.npy')
x_train = x_train[:Ntrain]

print('x_train shape:', x_train.shape)

########################################################
# Trainigsdatensatz umstellen um mu und sigma zu Lernen

'''
x_train_delta_list = SDE_Tools.make_SDE_training_data(x_train)
x_train_list = np.array([x_train_value_list, x_train_delta_list])
print('new train shape:',x_train_list.shape, x_train_value_list.shape)
x_train_list = np.transpose(x_train_list, [1,0,2])
print('new train shape:',x_train_list.shape, x_train_value_list.shape)
#print(x_train_list[0:51,:,:])
'''
x_train_derivatives = SDE_Tools.make_derivatives(x_train, M, frames, frames/T)
print('new train shape:', x_train_derivatives.shape)


########################################################
# Model trainieren
derivatives = SDE_Tools.make_tensorwise_derivatives(M, frames, fps)
ms = SDE_Tools.mu_sig_Net(M, latent_dim, n, akt_fun, complexity, forceHigherOrder=forceHigherOrder)

p_loss = SDE_Tools.make_pointwise_Loss(M, latent_dim, T, frames, ms, expected_SDE_complexity)
cv_loss = SDE_Tools.make_covariance_Loss(latent_dim, T, frames, batch_size, ms, expected_SDE_complexity)#, norm=lambda x: tf.math.sqrt(abs(x)))
ss_loss = SDE_Tools.make_sigma_size_Loss(latent_dim, ms)

reconstructor = SDE_Tools.make_Tensorwise_Reconstructor(d, latent_dim, n, T, frames, ms, expected_SDE_complexity)
rec_loss = SDE_Tools.make_reconstruction_Loss(M, n, T, frames, batch_size, reconstructor, derivatives)

#loss = lambda x_org,ms_rec: 1*rec_loss(x_org,None) + 1*p_loss(x_org, ms_rec) + 0.1*cv_loss(x_org, ms_rec) + 0*ss_loss(x_org, ms_rec)


def loss(x_org, ms_rec):
    S = 0
    #S += 1*rec_loss(x_org,None)
    S += 10*p_loss(x_org, ms_rec)
    S += 1*cv_loss(x_org, ms_rec)
    #S += 1000*ss_loss(x_org, ms_rec)
    return S


# , metrics=[lambda x,m: rec_loss(x[:,0,:])])
ms.compile(optimizer='adam', loss=loss, metrics=[cv_loss, lambda x, m: rec_loss(x, None)])
ms.fit(x_train_derivatives, x_train_derivatives,
       epochs=epochs, batch_size=batch_size, shuffle=False)
ms.summary()


########################################################
# Ergebnisse plotten

#fig, axs = plt.subplots(3, 5)

'''
x = np.linspace(0,T,frames)
x_2 = np.array([[x,np.zeros(frames)]])
x_2 = np.transpose(x_2, [0,2,1])
#print('predicting:',x_2.shape)
#y = ms.predict(x_2)
#print('predicted:',y.shape)

#axs[0, 1].plot(x,y[0,:,:,0])
axs[0, 1].set_title('Rekonstr.-mu')

a = np.array(list(map(mu, x_2[0,:,:])))
axs[0, 0].plot(x,a)
axs[0, 0].set_title('Original-mu')


sig_sq = np.array(list(map(lambda j: np.matmul(y[0,j,:,1:], np.transpose(y[0,j,:,1:],(1,0))), range(len(y[0])))))
#print('Shape5:',sig_sq.shape)
axs[1, 3].plot(x,sig_sq[:,0,:])
axs[1, 3].set_title('Rekonstr.-sig1')
axs[1, 3].axis((0,T,0,0.4))
axs[1, 4].plot(x,sig_sq[:,1,:])
axs[1, 4].set_title('Rekonstr.-sig2')
axs[1, 4].axis((0,T,0,0.4))
'''
'''
a = np.array(list(map(sigma, x_2[0,:,:])))
#print(a[0,:,:])
sig_sq = np.array(list(map(lambda j: np.matmul(a[j,:,:], np.transpose(a[j,:,:],(1,0))), range(len(a)))))
#print(sig_sq[0,:,:])
axs[1, 0].plot(x,sig_sq[:,0,:])
axs[1, 0].set_title('Original-sig1')
axs[1, 0].axis((0,T,0,0.4))
axs[1, 1].plot(x,sig_sq[:,1,:])
axs[1, 1].set_title('Original-sig2')
axs[1, 1].axis((0,T,0,0.4))
'''

xl = np.linspace(0, T, frames)
NrRec = 8
fig, axs = plt.subplots(2, NrRec)

x0 = x_train_derivatives[0:NrRec, 0, :, :]
# x0 hat dim: NrRec x M x latent_dim

R = reconstructor(x0)
print('reconstructed:', R.shape)

for i in range(NrRec):
    axs[0, i].plot(xl, x_train[i, :, :3])
    axs[1, i].plot(xl, R[i, :, 0, :3])

'''
#R = SDE_Tools.Reconstructor(d, n, T, frames, simulated_frames, np.zeros((NrRec,d)), ms, NrRec, applyBM=False)
axs[2, 0].plot(xl,R[0,:,:])
axs[2, 0].axis((0,T,-0.5,3))
axs[2, 0].set_title('SDE ohne BB (rec)')

for i in range(4):
    axs[2, i+1].plot(x,x_train[i,:,:])
    axs[2, i+1].axis((0,T,-0.5,3))
    axs[2, i+1].set_title('Ziehung der SDE')
'''


plt.show()
