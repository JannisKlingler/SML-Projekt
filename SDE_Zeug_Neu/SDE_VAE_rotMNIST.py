import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy as sp
from keras import backend as K

try:
    import SDE_Tools
    import AE_Tools
    import SDE_VAE_Tools
except:
    raise Exception('Could not load necessary Tools. Please execute file in its original location.')


# Needed for gpu support on some machines
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


########################################################
# Hyperparameter

# Hier bitte Pfad (absolut) angeben wo die Datensätze gespeichert werden sollen
# Ordner muss existieren, da Python auf vielen Systemen keine Ordner erstellen darf
#Bsp: data_path = 'C:/Users/[Name]/Desktop/Datasets/'
data_path = 'C:/Users/bende/Documents/Uni/Datasets/'


latent_dim = 15  # Dimension des latenten Raums (d)
frames = 20  # Anzahl der Frames im Datensatz (m+1)
M = 2  # Ordnung der SDEs
# Achtung: die letzten (M-1) frames können nicht zum trainieren verwendet werden
# Je chaotischer die Daten, desto größer N und kleiner M>1

# gibt an ob die Daten geglättet rekonstruiert werden sollen, oder ob man mit Brownschen Bewegungen neue Daten generieren will
reconstructWithBM = False
# gibt an ob wie beim ODE-2-VAE die Form einer ODE M-ten Ordnung erzwungen werden soll oder nicht
forceHigherOrder = False
# Faktor um die Komplexität der Netzwerke, welche die SDE lernen, zu bestimmen.
SDE_Net_complexity = 8*latent_dim
# [SDE_Net_complexity] sollte proportional zur latenten Dimension gewählt werden.

epochs = 10  # Anzahl der Epochen beim Haupt-Training
VAE_epochs_starting = 15  # Anzahl der Epochen beim vor-Training der En-&Decoder
# Anzahl der Epochen beim vor-Training der SDE-Netzwerke (geht viel schneller)
SDE_epochs_starting = 20
batch_size = 100
train_size = 60000  # <=60000
test_size = 100  # <=10000
act_CNN = 'relu'  # Aktivierungsfunktion für En-&Decoder
act_ms_Net = 'tanh'  # Aktivierungsfunktion für SDE-Netzwerke


# wenn diese parameter verändert werden, müssen die Datensätze neu erstellt werden

Time = 50  # SDEs werden besser gelernt, wenn [Time] ungefähr gleich [frames] ist.
fps = Time/frames  # ist in der Theorie gleich 1/(Delta t)
n = 1  # Anzahl der Brownschen Bewegungen in der SDE

# Falls die SDEs zu sehr schwanken um gut gelernt zu werden, kann dieser Wert höher gestellt werden.
D_t = 1


# Bitte nicht ändern:
pictureWidth = 28
pictureHeight = 28
pictureColors = 1



########################################################
# Datensatz laden oder erstellen
try:
    #raise Exception('Ich will den Datensatz neu erstellen')
    x_train = np.load(data_path+'rotatingMNIST_train_{}frames.npy'.format(frames))
    x_test = np.load(data_path+'rotatingMNIST_test_{}frames.npy'.format(frames))
except:
    print('Dataset is being generated. This may take a few minutes.')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_train))
    x_test_rot = list(map(lambda b: list(map(lambda i: np.where(sp.ndimage.rotate(
        b, (i+1) * 360/frames, reshape=False) > 127.5, 1.0, 0.0).astype('float32'), range(frames))), x_test))
    for j in range(len(x_test_rot)):
        for i in np.random.choice(range(3, 10), 3, replace=False):
            x_test_rot[j][i] = np.zeros((28, 28))
    x_train = np.transpose(np.array(x_train_rot), [0, 2, 3, 1])
    x_test = np.transpose(np.array(x_test_rot), [0, 2, 3, 1])
    try:
        np.save(data_path+'rotatingMNIST_train_{}frames.npy'.format(frames), x_train)
        np.save(data_path+'rotatingMNIST_test_{}frames.npy'.format(frames), x_test)
    except:
        print('could not save Dataset')
    print('Dataset generated')

x_train = x_train[0:train_size]
x_test = x_test[0:test_size]

# Dim: train_size x frames x pictureWidth x pictureHeight x pictureColors
x_train = np.transpose(np.array([x_train]), (1, 4, 2, 3, 0))
print('train-shape:', x_train.shape)


# Dim: test_size x frames x pictureWidth x pictureHeight x pictureColors
x_test = np.transpose(np.array([x_test]), (1, 4, 2, 3, 0))



########################################################
# Definitionen

derivatives = SDE_Tools.make_tensorwise_derivatives(M, frames, fps)

#encoder = AE_Tools.FramewiseEncoder(latent_dim, pictureWidth, pictureHeight, pictureColors, act_CNN, complexity=CNN_complexity, variational=True)
#decoder = AE_Tools.FramewiseDecoder(latent_dim, pictureWidth, pictureHeight, pictureColors, act_CNN, complexity=CNN_complexity)

encoder = AE_Tools.make_MNIST_encoder(latent_dim)
decoder = AE_Tools.make_MNIST_decoder(latent_dim)

ms_Net = SDE_Tools.mu_sig_Net(M, latent_dim, n, act_ms_Net,
                              SDE_Net_complexity, forceHigherOrder=forceHigherOrder)
reconstructor = SDE_Tools.Tensorwise_Reconstructor(
    latent_dim*pictureColors, n, Time, frames-M+1, ms_Net, D_t)


rec_loss = AE_Tools.make_binary_crossentropy_rec_loss(frames)
lr_loss = SDE_Tools.make_reconstruction_Loss(
    M, n, Time, frames, batch_size, reconstructor, derivatives)
p_loss = SDE_Tools.make_pointwise_Loss(M, latent_dim, Time, frames, ms_Net, D_t)
cv_loss = SDE_Tools.make_covariance_Loss(
    latent_dim, Time, frames, batch_size, ms_Net, D_t)
ss_loss = SDE_Tools.make_sigma_size_Loss(latent_dim, ms_Net)


def VAELoss(X_org, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List):
    S = 20*rec_loss(X_org, X_rec_List)
    return S


def SDELoss(Z_derivatives, ms_rec):
    S = 0
    S += 4*lr_loss(Z_derivatives, None)
    S += 10*p_loss(Z_derivatives, ms_rec)
    S += 0.5*cv_loss(Z_derivatives, ms_rec)
    return S


alpha = 0.5

def StartingLoss(X_org, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List):
    S = 20*rec_loss(X_org, X_rec_List)
    S += 5*lr_loss(Z_derivatives, Z_rec_List)
    S += alpha*10*p_loss(Z_derivatives,None)
    S += alpha*0.5*cv_loss(Z_derivatives,None)
    return S


'''
#Falls man am Ende nochmal En-&Decoder zusammen mit dem SDE-Netz trainieren will
def FullLoss(X_org, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List):
    S = 20*rec_loss(X_org, X_rec_List)
    S += 1*lr_loss(Z_derivatives, Z_rec_List)
    S += 10*p_loss(Z_derivatives, None)
    S += 1*cv_loss(Z_derivatives, None)
    return S
'''

########################################################
# SDE_VAE definieren

SDE_VAE = SDE_VAE_Tools.SDE_Variational_Autoencoder(
    M, 1, encoder, derivatives, reconstructor, decoder, StartingLoss)
# inp hat dim: None x frames x pictureWidth x pictureHeight x pictureColors

print('model defined')


########################################################
# Model ohne SDE-Rekonstruktion die latenten Darstellungen lernen lassen
print('initial training for encoder and decoder to learn a latent representation')
SDE_VAE.apply_reconstructor = False
SDE_VAE.compile(optimizer='adam', loss=lambda x, arg: arg)
SDE_VAE.fit(x_train, x_train, epochs=VAE_epochs_starting, batch_size=batch_size, shuffle=False)
SDE_VAE.summary()

'''
# Latente Darstellung zum testen abspeichern
_,_,Z_enc,_,_,_ = SDE_VAE.fullcall(x_train)
np.save(data_path+'TestIfEncoderWorks',Z_enc)
'''

########################################################
# Die SDE-Rekonstruktion der latenten Darstellungen lernen lassen
# Dieses training ist merklich schneller auf der haupt-cpu ohne verwendung einer gpu
print('training to learn SDE governing latent representation')

new_ms_Net = SDE_Tools.mu_sig_Net(M, latent_dim, n, act_ms_Net, SDE_Net_complexity, forceHigherOrder=forceHigherOrder)
new_ms_Net.compile(optimizer='adam', loss=SDELoss, metrics=[
               ss_loss, lambda x, m: lr_loss(x, None)])
reconstructor.ms_Net = new_ms_Net

with tf.device('/cpu:0'):
    new_ms_Net.compile(optimizer='adam', loss=SDELoss, metrics=[
                   ss_loss, lambda x, m: lr_loss(x, None)])
    _,_,Z_enc,_,_,_ = SDE_VAE.fullcall(x_train)
    z_train_derivatives = tf.constant(derivatives(Z_enc))
    new_ms_Net.fit(z_train_derivatives, z_train_derivatives,
               epochs=SDE_epochs_starting, batch_size=batch_size, shuffle=False)
    new_ms_Net.summary()


'''
########################################################
# En-&Decoder und SDE-Rekonstruktion zusammen trainieren
# Optional. Manchmal sehen Rekonstructionen damit besser aus
print('main training with SDEs and Decoders combined')
SDE_VAE.custom_loss = FullLoss
SDE_VAE.compile(optimizer='adam', loss=lambda x, arg: arg)
SDE_VAE.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=False)
SDE_VAE.summary()
'''

########################################################
# Modell so einstellen, dass glatter reconstruiert wird.
SDE_VAE.apply_reconstructor = True
reconstructor.applyBM = reconstructWithBM


########################################################
# Ergebnisse speichern
_, _, Z_enc_List, _, Z_rec_List, X_rec_List = SDE_VAE.fullcall(x_test)

#np.save(data_path+'Results_SDE_rotMNIST_Z_org_{}frames'.format(frames), x_test_path)
np.save(data_path+'Results_SDE_rotMNIST_X_org_{}frames'.format(frames), x_test)
np.save(data_path+'Results_SDE_rotMNIST_Z_enc_{}frames'.format(frames), Z_enc_List)
np.save(data_path+'Results_SDE_rotMNIST_Z_rec_{}frames'.format(frames), Z_rec_List)
np.save(data_path+'Results_SDE_rotMNIST_X_rec_{}frames'.format(frames), X_rec_List)


########################################################
# Ergebnisse darstellen

_, _, enc_lat, _, rec_lat, rec_imgs = SDE_VAE.fullcall(x_test)

#print('enc_lat:', enc_lat.shape)
#print('rec_lat:', rec_lat.shape)

fig, axs = plt.subplots(9, 10)
for i in range(4):
    for j in range(10):
        axs[2*i, j].imshow(x_test_org[i, j, :, :, 0], cmap='gray')
        axs[2*i+1, j].imshow(rec_imgs[i, j, :, :, 0], cmap='gray')
for i in range(5):
    axs[8, 2*i].plot(np.linspace(1, frames, frames), enc_lat[i, :, :], '-')
    axs[8, 2*i+1].plot(np.linspace(1, frames-M+1, frames-M+1), rec_lat[i, :, :], '-')
plt.show()
