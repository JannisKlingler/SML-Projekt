import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from math import exp, pi


from mpl_toolkits.axes_grid1 import ImageGrid
from math import sin, cos, sqrt, pi, floor, ceil, exp, log
from random import choice, uniform


frames = 100
T = 1
fps = frames//T

SDE_List = [(lambda t,x : 1, lambda t,x : t)]

#Datensatzgröße
Ntrain = 5000
Ntest = 1000



# T = Zeit, die simuliert wird
# D = Anzahl der simulierten schritte pro Sekunde
#X_0 = Startwert
def SDE(T,D,X_0,mu_sigma):
    mu, sigma = mu_sigma
    N = ceil((T+1)*D)
    E = sqrt(3/D)
    Y = np.random.uniform(-E,E,N)
    #Y = list(map(lambda i: uniform(-E,E), range(0,N)))
    X = list(map(lambda arg: X_0,range(fps)))
    for i in range(0,N):
        X.append( X[-1] + mu(i/D, X[-1])/D + sigma(i/D, X[-1]) * Y[i])
    return X

K_rbf = lambda y : exp(-y**2)/sqrt(pi)
z = 0.5     #Zwischen 1/fps und 1
K_historyAverage = lambda y : 1/z if (y > -z and y <= 0) else 0

def smooth(X, h, K):
    Xnew = []
    for i in range(len(X)):
        Xnew.append(sum(map(lambda j: (1/fps)*X[j]*K((i-j)/fps/h)/h, range(len(X)))))
    return Xnew



x_train = list(map(lambda i : SDE(T, fps, 1, choice(SDE_List)), range(Ntrain)))
x_test = list(map(lambda i : SDE(T, fps, 1, choice(SDE_List)), range(Ntest)))



##########################
#Zeichen
fig, axs = plt.subplots(6, 6)

for i in range(18):
    j,k = i//6 , i%6
    axs[2*j, k].plot(list(np.linspace(0,T,frames)),x_train[i][fps:(T+1)*fps])
    axs[2*j, k].axis((0,T,-1,3))
    axs[2*j+1, k].plot(list(np.linspace(0,T,frames)),smooth(x_train[i], 0.2, K_historyAverage)[fps:(T+1)*fps])
    axs[2*j+1, k].axis((0,T,-1,3))
#plt.draw()
plt.show()
