import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import tensorflow as tf
from tensorflow import keras


from mpl_toolkits.axes_grid1 import ImageGrid
from math import sin, cos, sqrt, pi, floor, ceil, exp, log
from random import choice, uniform

# T = Zeit, die simuliert wird
# D = Anzahl der simulierten schritte pro Sekunde
#X_0 = Startwert
def SDE(T,D,X_0,mu_sigma):
    mu, sigma = mu_sigma
    N = ceil(T*D)
    E = sqrt(3/D)
    Y = np.random.uniform(-E,E,N)
    #Y = list(map(lambda i: uniform(-E,E), range(0,N)))
    X = [X_0]
    for i in range(0,N):
        X.append( X[-1] + mu(i/D, X[-1])/D + sigma(i/D, X[-1]) * Y[i])
    return X


frames = 100
T = 1

SDE_List = [(lambda t,x : t, lambda t,x : 0), (lambda t,x : 0, lambda t,x : 1)]

#Datensatzgröße
Ntrain = 5000
Ntest = 1000


x_train = list(map(lambda i : SDE(T, frames/T, 1, choice(SDE_List)), range(Ntrain)))
x_test = list(map(lambda i : SDE(T, frames/T, 1, choice(SDE_List)), range(Ntest)))



##########################
#Zeichen
fig, axs = plt.subplots(5, 5)

for i in range(25):
    j,k = i//5 , i%5
    axs[j, k].plot(list(np.linspace(0,T,frames+1)),x_train[i])
    axs[j, k].axis((0,T,-1,3))
#plt.draw()
plt.show()
