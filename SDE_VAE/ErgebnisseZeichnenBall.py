#import datasets as data
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import SDE_Tools
import AE_Tools
from math import pi

frames = 50

X_0 = np.array([0,1])
# mu : R^d -> R^d , für d = latent_dim
def mu(x):
    m = np.array([x[1], -x[0]])
    return m

# sigma: R^d -> R^(dxn) , , für d = latent_dim
def sigma(x):
    #s = np.array([[0.2], [0.1]])
    s = np.zeros((2,1))
    return s

Z_org_smooth = np.array(SDE_Tools.ItoDiffusion(2, 1, 3*pi, frames, frames, X_0, mu, sigma))

data_path = 'C:/Users/bende/Documents/Uni/Datasets/'

Z_org = np.load(data_path+'Results_SDE_Ball_Z_org_{}frames.npy'.format(frames))
X_org = np.load(data_path+'Results_SDE_Ball_X_org_{}frames.npy'.format(frames))

Z_enc = np.load(data_path+'Results_SDE_Ball_Z_enc_{}frames.npy'.format(frames))
Z_rec = np.load(data_path+'Results_SDE_Ball_Z_rec_{}frames.npy'.format(frames))
Z_recBM = np.load(data_path+'Results_SDE_Ball_Z_recBM_{}frames.npy'.format(frames))
X_rec = np.load(data_path+'Results_SDE_Ball_X_rec_{}frames.npy'.format(frames))

print('shape:',Z_rec.shape)

'''
fig, axs = plt.subplots(1, 2)

xl = np.linspace(0,frames,frames)

axs[0].plot(xl, -Z_org_smooth[:,0])
axs[1].plot(xl, Z_rec[0,:frames,0])
'''

'''
A = 5
fig, axs = plt.subplots(2, A)
xl = np.linspace(0,frames,frames)
for i in range(A):
    axs[0,i].plot(xl, -Z_org[i,:frames,0])
    axs[1,i].plot(xl, Z_enc[i,:frames,0])
'''

'''
fig, axs = plt.subplots(8, 11)
for i in range(8):
    for j in range(11):
        axs[i,j].axis('off')
for i in range(3):
    for j in range(5):
        axs[3*i,j].imshow(X_org[i,j,:,:,0], cmap='gray')
        axs[3*i+1,j].imshow(X_rec[i,j,:,:,0], cmap='gray')
    for j in range(5):
        axs[3*i,6+j].imshow(X_org[i,-5+j,:,:,0], cmap='gray')
        axs[3*i+1,6+j].imshow(X_rec[i,-5+j,:,:,0], cmap='gray')
'''

'''
xl = np.linspace(0,frames,frames)
fig, axs = plt.subplots(1, 4)
axs[0].plot(xl, Z_org_smooth[:frames,0])
axs[0].set_title('ohne Unsicherheit')
for i in range(3):
    axs[i+1].plot(xl, -Z_enc[i,:frames])
    axs[i+1].set_title('Ziehung mit Unsicherheit')
'''

'''
xl = np.linspace(0,frames,frames)
fig, axs = plt.subplots(1, 4)
axs[0].plot(xl, Z_rec[:frames,0])
axs[0].set_title('ohne Unsicherheit')
for i in range(3):
    axs[i+1].plot(xl, -Z_enc[i,:frames])
    axs[i+1].set_title('Ziehung mit Unsicherheit')
'''


fig, axs = plt.subplots(1, 1)
xl = np.linspace(0,3*pi,frames)
axs.plot(xl, Z_rec[0,:,0],linewidth=3)
axs.axis([0, 3*pi, -1.5, 1.5])
#axs[i].set_title('m={}'.format(m))
axs.axis('off')


plt.show()
