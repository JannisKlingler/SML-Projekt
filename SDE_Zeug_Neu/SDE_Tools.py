#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from math import sin, cos, sqrt, pi, floor, ceil, exp, log
from keras import backend as K

#tf.random.set_seed(1)





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
def Reconstructor(d, n, T, frames, simulated_frames, X_0_List, ms_Net, NrReconstr, applyBM=False):
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
        #print('mult_sig_Y:',mult_sig_Y.shape)
        X_sim.append( X_sim[-1] + mu/fps + mult_sig_Y)
    X = []
    for i in range(frames):
        X.append(X_sim[floor(i*simulated_frames/frames)])
    X = np.array(X)
    X = np.transpose(X,(1,0,2))
    #print('reconstructed:',X.shape)
    return X


def make_Tensorwise_Reconstructor(d, n, T, frames, ms_Net, batchSize, simulated_frames=False, applyBM=False):
    if not simulated_frames:
        simulated_frames = frames
    fps = simulated_frames/T
    E = sqrt(3/fps)
    def T_Reconstructor(X_0_List):
        print('reconstructing form X_0:',X_0_List.shape)
        X_sim = [X_0_List]
        for i in range(simulated_frames):
            #mu_sig = tf.map_fn(lambda x: ms_Net.eval(x),X_sim[-1])
            #print('CCC0:',tf.transpose([X_sim[-1]],[0,1,2]))
            mu = ms_Net(tf.transpose([X_sim[-1]],[1,0,2]))[0,:,:,0]
            #print('CCC:',mu.shape)
            X_sim.append(X_sim[-1] + mu/fps)
        X = []
        for i in range(frames):
            X.append(X_sim[floor(i*simulated_frames/frames)])
        X = tf.stack(X, axis=1)
        #print('rec:',X)
        return X

    return T_Reconstructor



########################################################
#Trainigsdatensatz umstellen um mu und sigma zu Lernen

#Input braucht Form: Ntrain x frames x latent_dim
#Output hat Form: 2 x (Ntrain*(frames-1)) x latent_dim
def make_SDE_training_data(x_train):
    x_train = np.array(x_train)
    Ntrain, frames, d = x_train.shape
    L = list(map(lambda i: x_train[i,1:,:]-x_train[i,0:-1,:] ,range(Ntrain)))
    #x_train_delta_list = np.concatenate(L,axis=0)
    x_train_delta_list = np.stack(L,axis=0)

    return x_train_delta_list



########################################################
#Netzwerk zum lernen der SDE aufbauen
class mu_sig_Net(tf.keras.Model):
    def __init__(self, d, n, akt_fun, complexity):
        super(mu_sig_Net, self).__init__()

        #self.inp = z = tf.keras.layers.Input(shape=(d))

        #Netzwerk um mu zu lernen
        self.mu_layer1 = tf.keras.layers.Dense(complexity*d, activation=akt_fun)
        self.mu_layer2 = tf.keras.layers.Dense(complexity*d, activation=akt_fun)
        self.mu_layer3 = tf.keras.layers.Dense(d, activation=akt_fun)
        self.mu_layer4 = tf.keras.layers.Reshape((d, 1))

        #Netzwerk um sigma zu lernen
        self.sig_layer1 = tf.keras.layers.Dense(complexity*d*n, activation=akt_fun)
        self.sig_layer2 = tf.keras.layers.Dense(complexity*d*n, activation=akt_fun)
        self.sig_layer3 = tf.keras.layers.Dense(d*n, activation=akt_fun)
        self.sig_layer4 = tf.keras.layers.Reshape((d, n))

        #Netzwerke zusammenfügen
        #self.merge_layer = tf.keras.layers.Concatenate(axis=-1)([mu,sig])

        #super(mu_sig_Net, self).__init__(self.inp, mu_sig, name="mu_sig_Net")

    def call(self, inputs):
        print('called ms_Net on:',inputs.shape)
        frames_List = tf.split(inputs, len(inputs[0]), axis=1)
        #print('changed:',frames_List)

        ms_List = []
        for x in frames_List:
            #print(x[:,0,:])
            mu = self.mu_layer1(x[:,0,:])
            mu = self.mu_layer2(mu)
            mu = self.mu_layer3(mu)
            mu = self.mu_layer4(mu)
            sig = self.sig_layer1(x[:,0,:])
            sig = self.sig_layer2(sig)
            sig = self.sig_layer3(sig)
            sig = self.sig_layer4(sig)
            mu_sig = tf.keras.layers.Concatenate(axis=-1)([mu,sig])
            ms_List.append(mu_sig)
        ms_List = tf.stack(ms_List, axis=1)
        #print('ms_Net output:',ms_List.shape)
        return ms_List
'''
    def mu(self, x):
        x1 = self.mu_layer1(x)
        x2 = self.mu_layer2(x1)
        x3 = self.mu_layer3(x2)
        return x3

    def sig(self, x):
        sig = self.sig_layer1(x)
        sig = self.sig_layer2(sig)
        sig = self.sig_layer3(sig)
        sig = self.sig_layer4(sig)
        return sig
'''

########################################################
#Verlustfunktion definieren

'''
def reconstruction_loss(X_org, X_rec):
    Diff = X_org - X_rec
    Diff = tf.map_fn(norm1,Diff)
    Diff = K.sum(Diff)
    return Diff
'''

def make_reconstruction_Loss(d, n, Time, frames, batch_size, T_reconstructor, norm=abs):
    def reconstruction_loss(X_org):
        X_org = tf.split(X_org, batch_size, axis=0)
        X_org = tf.stack(X_org, axis=0)
        X_0 = X_org[:,0,:]

        X_rec = T_reconstructor(X_0)
        Diff = X_org - X_rec
        Diff = tf.map_fn(norm,Diff)
        Diff = K.sum(Diff)
        return Diff
    return reconstruction_loss


def make_pointwise_Loss(Time, frames, ms_Net, Model, norm=abs, getOwnInputs=False):
    def pointwise_loss(X_org):
        if getOwnInputs:
            X_org,_,_ = Model.fullcall(X_org)
        print('p_loss-inp:', X_org.shape)
        #mu = mu_sig[:,:,0]
        X_delta = X_org[:,1:,:]-X_org[:,:-1,:]
        mu = ms_Net(X_org)[:,:-1,:,0]
        fps = frames/Time
        rec_delta = mu/fps
        Difference = X_delta - rec_delta
        Diff = tf.map_fn(norm,Difference)
        Diff = K.mean(Diff)
        return Diff

    return pointwise_loss

def make_covariance_Loss(Time, frames, batch_size, ms_Net, Model, norm=abs, getOwnInputs=False):
    def covariance_loss(X_org):
        if getOwnInputs:
            X_org,_,_ = Model.fullcall(X_org)
        print('cv_loss-inp:', X_org.shape)
        X_value = X_org[:,:-1,:]
        #print('X_value:',X_value.shape)
        X_delta = X_org[:,1:,:]-X_org[:,:-1,:]
        #print('X_delta:',X_delta.shape)
        mu_sig = ms_Net(X_value)
        #print('ms:',mu_sig.shape)
        mu = mu_sig[:,:,:,0]
        sig = mu_sig[:,:,:,1:]

        fps = frames/Time
        rec_delta = mu/fps
        random_delta = [X_delta - rec_delta]
        random_delta = tf.transpose(random_delta, [1,2,3,0])
        print('random_delta:',random_delta.shape)
        covar = list(map(lambda i: tf.keras.layers.Dot(axes=2)([random_delta[:,i,:,:],random_delta[:,i,:,:]]), range(frames-1)))
        covar = tf.stack(covar, axis=1)
        #print('covar:',covar.shape)
        covar = K.sum(covar,axis=1)
        #print('covar:',covar.shape)

        #print('sig:',sig.shape)
        sum_sig = list(map(lambda i: tf.keras.layers.Dot(axes=2)([sig[:,i,:,:],sig[:,i,:,:]]), range(frames-1)))
        sum_sig = tf.stack(sum_sig, axis=1)
        #print('A:',sum_sig)
        theoretical_covar = K.sum(sum_sig, axis=1)/fps
        #print('theoretical_covar:',theoretical_covar.shape)

        Diff_sq_var = covar - theoretical_covar
        Diff_sq_var = tf.map_fn(norm,Diff_sq_var)
        #print('Diff_sq_var:',Diff_sq_var.shape)
        Diff_sq_var = K.mean(Diff_sq_var)
        return Diff_sq_var

    return covariance_loss


def make_sigma_size_Loss(ms_Net, Model, norm=abs, getOwnInputs=False):
    def sigma_size_loss(X_org):
        if getOwnInputs:
            X_org,_,_ = Model.fullcall(X_org)
        X_value = X_org[:,:-1,:]
        sig = ms_Net(X_value)[:,:,:,1:]
        sig_size = tf.map_fn(norm,sig)
        sig_size = K.mean(sig_size)
        return sig_size

    return sigma_size_loss
