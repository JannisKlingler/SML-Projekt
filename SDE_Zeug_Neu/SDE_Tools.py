#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from math import sin, cos, sqrt, pi, floor, ceil, exp, log
from keras import backend as K

#tf.random.set_seed(1)

#X_org hat dim: None x frames x latent_dim
#Output hat dim: None x frames-M+1 x M x latent_dim
def make_derivatives(x_org, M, frames, fps):
    dList = [x_org]
    for i in range(1,M):
        x_last = dList[-1]
        x_derived = x_last[:,1:,:]-x_last[:,:-1,:]
        x_derived = x_derived*fps
        #print('ZZZZ', x_derived.shape)
        dList.append(x_derived)
    dList = list(map(lambda arg: arg[:,:frames-M+1,:],dList))
    return np.stack(dList,axis=2)

#X_org hat dim: None x frames x latent_dim
#Output hat dim: None x frames-M+1 x M x latent_dim
def make_tensorwise_derivatives(M, frames, fps):
    def derivatives(x_org):
        length = x_org.shape[1]
        dList = [x_org]
        for i in range(1,M):
            x_last = dList[-1]
            x_derived = x_last[:,1:,:]-x_last[:,:-1,:]
            x_derived = x_derived*fps
            #print('ZZZZ', x_derived.shape)
            dList.append(x_derived)
        dList = list(map(lambda arg: arg[:,:length-M+1,:],dList))
        return tf.stack(dList,axis=2)
    return derivatives



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
'''
#rekonstruiere den Pfad nach den gelernten mu,sigma (Parallel)
#Output-Dim : NrReconstr x frames x d
#X_0 hat Form: NrReconstr x d
def Reconstructor(d, latent_dim, n, T, frames, simulated_frames, X_0_List, ms_Net, NrReconstr, applyBM=False):
    fps = simulated_frames/T
    E = sqrt(3/fps)
    if applyBM:
        Y = np.random.uniform(-E,E,(NrReconstr,simulated_frames,n))
    else:
        Y = np.zeros((NrReconstr,simulated_frames,n))
    X_sim = [X_0_List]
    for i in range(simulated_frames):
        ms = ms_Net.predict(X_sim[-1])
        mu = ms[:,:latent_dim,0]
        sig = ms[:,:latent_dim,1:]
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
'''

def make_Tensorwise_Reconstructor(d, latent_dim, n, T, frames, ms_Net, batchSize, simulated_frames=False, applyBM=False):
    if not simulated_frames:
        simulated_frames = frames
    fps = simulated_frames/T
    E = sqrt(3/fps)
    def T_Reconstructor(X_0_List):
        print('reconstruction from:',X_0_List.shape)
        #X_0_List hat dim: None x M x latent_dim
        X_sim = [X_0_List]
        for i in range(simulated_frames):
            mu = ms_Net(tf.transpose([X_sim[-1]],[1,0,2,3]))[:,0,:,:,0]
            #mu hat dim: None x M x latent_dim
            X_sim.append(X_sim[-1] + mu/fps)
        X = []
        for i in range(frames):
            X.append(X_sim[floor(i*simulated_frames/frames)])
        X = tf.stack(X, axis=1)
        #X hat dim: None x frames x M x latent_dim
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
    def __init__(self, M, latent_dim, n, akt_fun, complexity, forceHigherOrder=False):
        super(mu_sig_Net, self).__init__()
        self.latent_dim = latent_dim
        self.M = M
        self.forceHigherOrder = forceHigherOrder
        d = M*latent_dim
        #self.inp = z = tf.keras.layers.Input(shape=(latent_dim))
        self.mu_layerList = []
        self.sig_layerList = []

        #Netzwerk um mu zu lernen
        if forceHigherOrder:
            self.mu_layerList.append(tf.keras.layers.Flatten())
            self.mu_layerList.append(tf.keras.layers.Dense(complexity, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Dense(complexity, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Dense(latent_dim, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Reshape((1,latent_dim)))
        else:
            self.mu_layerList.append(tf.keras.layers.Flatten())
            self.mu_layerList.append(tf.keras.layers.Dense(complexity*M, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Dense(complexity*M, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Dense(M*latent_dim, activation=akt_fun))
            self.mu_layerList.append(tf.keras.layers.Reshape((M,latent_dim)))

        #Netzwerk um sigma zu lernen
        self.sig_layerList.append(tf.keras.layers.Flatten())
        self.sig_layerList.append(tf.keras.layers.Dense(complexity*M*n, activation=akt_fun))
        self.sig_layerList.append(tf.keras.layers.Dense(complexity*M*n, activation=akt_fun))
        self.sig_layerList.append(tf.keras.layers.Dense(M*latent_dim*n, activation=akt_fun))
        self.sig_layerList.append(tf.keras.layers.Reshape((M, latent_dim, n)))

        #Netzwerke zusammenfügen
        #self.merge_layer = tf.keras.layers.Concatenate(axis=-1)([mu,sig])

        #super(mu_sig_Net, self).__init__(self.inp, mu_sig, name="mu_sig_Net")

    def call(self, inputs):
        #inputs hat dim: batch_size x frames-M+1 x M x latent_dim
        print('called ms_Net on:',inputs.shape)
        frames_List = tf.split(inputs, inputs.shape[1], axis=1)
        #print('changed:',frames_List)
        ms_List = []
        for x in frames_List:
            #x hat dim: batch_size x 1 x M x latent_dim

            mu = x[:,0,:,:]
            for l in self.mu_layerList:
                mu = l(mu)
            if self.forceHigherOrder:
                mu = tf.keras.layers.Concatenate(axis=1)([x[:,0,1:,:], mu])
            #mu hat dim: batch_size x M x latent_dim

            #eine dim an mu anhängen, um später mit sig concat machen zu können
            mu = tf.transpose([mu],(1,2,3,0))
            #mu hat dim: batch_size x M x latent_dim x 1

            sig = x[:,0,:,:]
            for l in self.sig_layerList:
                sig = l(sig)
            #sig hat dim: batch_size x M x latent_dim x n
            print('mu and sig:', mu.shape, sig.shape)
            mu_sig = tf.keras.layers.Concatenate(axis=-1)([mu,sig])
            #mu_sig hat dim: batch_size x M x latent_dim x 1+n

            ms_List.append(mu_sig)
        print('ms_Net output0:',len(ms_List),ms_List[0].shape)
        ms_List = tf.stack(ms_List, axis=1)
        print('ms_Net output:',ms_List.shape)
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

def make_reconstruction_Loss(M, n, Time, frames, batch_size, T_reconstructor, derivatives, norm=abs):
    def reconstruction_loss(Z_org, Z_rec):
        #X_org = tf.split(X_org, batch_size, axis=0)
        #X_org = tf.stack(X_org, axis=0)
        print('SDE_rec_loss_inp:',Z_org.shape)#, Z_rec.shape)
        if Z_rec is None:
            #Z_derivatives = derivatives(Z_org)
            #print('CCCC0:',Z_derivatives.shape)
            Z_0 = Z_org[:,0,:,:]
            #print('CCCC0:',Z_0.shape)
            Z_rec = T_reconstructor(Z_0)[:,:Z_org.shape[1],0,:]
            Z_org = Z_org[:,:,0,:]
        print('SDE_rec_loss_inp2:',Z_org.shape, Z_rec.shape)
        Diff = Z_org[:,:frames-M+1,:] - Z_rec
        Diff = tf.map_fn(norm,Diff)
        #Diff = tf.keras.layers.Multiply()([Diff,Diff])
        Diff = K.mean(Diff)
        return Diff
    return reconstruction_loss


def make_pointwise_Loss(M, latent_dim, Time, frames, ms_Net,norm=abs):
    def pointwise_loss(Z_org, ms_rec):
        print('p_loss-inp1:', Z_org.shape)
        if ms_rec is None:
            ms_rec = ms_Net(Z_org)
        print('p_loss-inp2:', ms_rec.shape)
        Z_delta = Z_org[:,1:,:,:] - Z_org[:,:-1,:,:]
        mu = ms_rec[:,:-1,:,:,0]
        fps = frames/Time
        rec_delta = mu/fps
        Difference = Z_delta - rec_delta
        #Diff = tf.map_fn(lambda arg1: tf.map_fn(lambda arg2: tf.map_fn(lambda arg3: tf.map_fn(abs,arg3),arg2),arg1),Difference)
        Diff = tf.map_fn(abs, Difference)
        Diff = K.mean(Diff)
        return Diff
    return pointwise_loss

def make_covariance_Loss(latent_dim, Time, frames, batch_size, ms_Net, norm=abs):
    def covariance_loss(Z_org, ms_rec):
        #Z_org hat dim: batch_size x frames-M+1 x M x latent_dim
        Z_delta = Z_org[:,1:,:,:]-Z_org[:,:-1,:,:]
        #Z_org hat dim: batch_size x frames-M x M x latent_dim
        if ms_rec is None:
            ms_rec = ms_Net(Z_org)
        #ms_rec hat dim: batch_size x frames-M+1 x M x latent_dim x 1+n
        mu = ms_rec[:,:,:,:,0]
        #mu hat dim: batch_size x frames-M+1 x M x latent_dim
        sig = ms_rec[:,:,:,:,1:]
        #sig hat dim: batch_size x frames-M+1 x M x latent_dim x n

        fps = frames/Time
        rec_delta = mu/fps
        random_delta = [Z_delta[:,:,0,:] - rec_delta[:,:-1,0,:]]
        #print('random_delta:',random_delta.shape)
        random_delta = tf.transpose(random_delta, [1,2,3,0])
        #random_delta hat dim: batch_size x frames[lokal] x latent_dim x 1
        covar = list(map(lambda i: tf.keras.layers.Dot(axes=2)([random_delta[:,i,:,:],random_delta[:,i,:,:]]), range(random_delta.shape[1])))
        covar = tf.stack(covar, axis=1)
        covar = K.sum(covar,axis=1)
        #covar hat dim: batch_size x frames[lokal] x latent_dim x latent_dim

        sig = sig[:,:,0,:,:]
        #sig hat dim: batch_size x frames[lokal] x latent_dim x n
        sum_sig = list(map(lambda i: tf.keras.layers.Dot(axes=2)([sig[:,i,:,:],sig[:,i,:,:]]), range(sig.shape[1])))
        sum_sig = tf.stack(sum_sig, axis=1)
        theoretical_covar = K.sum(sum_sig, axis=1)/fps
        #theoretical_covar hat dim: batch_size x frames[lokal] x latent_dim x latent_dim
        #print('theoretical_covar:',theoretical_covar.shape)

        Diff_sq_var = covar - theoretical_covar
        Diff_sq_var = tf.map_fn(norm,Diff_sq_var)
        #print('Diff_sq_var:',Diff_sq_var.shape)
        Diff_sq_var = K.mean(Diff_sq_var)/(latent_dim**2)
        return Diff_sq_var

    return covariance_loss


def make_sigma_size_Loss(latent_dim, ms_Net, norm=abs):
    def sigma_size_loss(Z_org, ms_rec):
        #Z_org hat dim: batch_size x frames-M+1 x M x latent_dim
        if ms_rec is None:
            ms_rec = ms_Net(Z_org)
        #ms_rec hat dim: batch_size x frames-M+1 x M x latent_dim x 1+n
        sig = ms_rec[:,:,0,:,1:]
        sig_size = tf.map_fn(norm,sig)
        sig_size = K.mean(sig_size)/latent_dim
        return sig_size

    return sigma_size_loss

def make_anti_regularizer_Loss(latent_dim, s_Net, Model, norm=abs):
    def anti_regularizer_loss(X_org):
        _,_,_,X_rec,_ = Model.fullcall(X_org)
        print('ar_loss X_rec:',X_rec.shape)
        one_variance = X_rec[:,1:,:] - X_rec[:,:-1,:]
        one_variance = tf.map_fn(abs, one_variance)
        one_variance = K.sum(one_variance, axis=1)
        print('one_variance:',one_variance.shape)
        loss = -tf.map_fn(abs, one_variance)

        avg_norm = tf.map_fn(abs, X_rec)
        avg_norm = K.mean(avg_norm)
        return K.mean(loss)/avg_norm

    return anti_regularizer_loss
