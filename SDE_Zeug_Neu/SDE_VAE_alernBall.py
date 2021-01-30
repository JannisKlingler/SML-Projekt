import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import draw
from keras import backend as K
from math import pi

try:
    import SDE_Tools
    import AE_Tools
    import SDE_VAE_Tools
except:
    raise Exception('Could not load necessary Tools. Please execute file in its original location.')


# Diesen Block einkommentieren, falls man Python auf der gpu laufen lässt

# Needed for gpu support on some machines
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


########################################################
# Hyperparameter

# Hier bitte Pfad (absolut) angeben wo die Datensätze gespeichert werden sollen
# Ordner muss existieren, da Python auf vielen Systemen keine Ordner erstellen darf
#Bsp: data_path = 'C:/Users/[Name]/Desktop/Datasets/'
data_path = 'C:/Users/Admin/Desktop/Python/Datasets/'

latent_dim = 1  # Dimension des latenten Raums (d)
frames = 50  # Anzahl der Frames im Datensatz (m+1)
M = 2  # Ordnung der SDEs
N = 1  # Anzahl der frames, über die gemittelt wird um Ableitungen zu approzimieren
# Achtung: die letzten (M-1)*N frames können nicht zum trainieren verwendet werden
# Je chaotischer die Daten, desto größer N und kleiner M>1

# gibt an ob die Daten geglättet rekonstruiert werden sollen, oder ob man mit Brownschen Bewegungen neue Daten generieren will
reconstructWithBM = False
# gibt an ob wie beim ODE-2-VAE die Form einer ODE M-ten Ordnung erzwungen werden soll oder nicht
forceHigherOrder = False
# Faktor um die Komplexität der Netzwerke, welche die SDE lernen, zu bestimmen.
SDE_Net_complexity = 20
# [SDE_Net_complexity] sollte proportional zur latenten Dimension gewählt werden.

epochs = 10  # Anzahl der Epochen beim Haupt-Training
VAE_epochs_starting = 5  # Anzahl der Epochen beim vor-Training der En-&Decoder
# Anzahl der Epochen beim vor-Training der SDE-Netzwerke (geht viel schneller)
SDE_epochs_starting = 5
batch_size = 50
train_size = 3000  # <3000
test_size = 100  # <1000
act_CNN = 'relu'  # Aktivierungsfunktion für En-&Decoder
act_ms_Net = 'tanh'  # Aktivierungsfunktion für SDE-Netzwerke


# wenn diese parameter verändert werden, müssen die Datensätze neu erstellt werden

Time = 50  # SDEs werden besser gelernt, wenn [Time] ungefähr gleich [frames] ist.
# Frames, die beim erstellen des Datensatzes simuliert werden. Das Programm sieht davon nur [frames] viele.
simulated_frames = 200
simulated_Time = 3*pi  # Zeit, die beim erstellen des Datensatzes simuliert wird.
fps = Time/frames  # ist in der Theorie gleich 1/(Delta t)
n = 1  # Anzahl der Brownschen Bewegungen in der SDE
# Falls die SDEs zu sehr schwanken um gut gelernt zu werden, kann dieser Wert höher gestellt werden.
expected_SDE_complexity = 1


# Bitte nicht ändern:
pictureWidth = 28
pictureHeight = 28
pictureColors = 1


################################################################################
# Datensatz laden oder erstellen

try:
    x_train = np.load(data_path+'SDE_Ball_train_{}frames.npy'.format(frames))
    x_test = np.load(data_path+'SDE_Ball_test_{}frames.npy'.format(frames))
    x_train_path = np.load(data_path+'SDE_Ball_train_path_{}frames.npy'.format(frames))
    x_test_path = np.load(data_path+'SDE_Ball_test_path_{}frames.npy'.format(frames))
    print('loaded datasets')
except:
    print('Dataset is being generated. This may take a few minutes.')
    X_0 = np.array([np.zeros(train_size), np.ones(train_size)])
    X_0 = np.transpose(X_0, [1, 0])

    # mu : R^d -> R^d , für d = latent_dim
    def mu(x):
        m = np.array([x[1], -x[0]])
        return m

    # sigma: R^d -> R^(dxn) , , für d = latent_dim
    def sigma(x):
        s = np.array([[0.2], [0.1]])
        #s = np.zeros((d,n))
        return s

    x_train_path = np.array(list(map(lambda i: SDE_Tools.ItoDiffusion(
        2, n, simulated_Time, frames, simulated_frames, X_0[i], mu, sigma), range(3000))))
    x_test_path = np.array(list(map(lambda i: SDE_Tools.ItoDiffusion(
        2, n, simulated_Time, frames, simulated_frames, X_0[i], mu, sigma), range(1000))))

    List = []
    for i in range(x_train_path.shape[0]):
        list = []
        for j in range(frames):
            position = [x_train_path[i, j, 0]/8+0.5, 0.5]
            radius = 0.12
            arr = np.zeros((28, 28))
            ro, co = draw.disk((position[0] * 28, position[1] *
                                28), radius=radius*28, shape=arr.shape)
            arr[ro, co] = 1
            list.append(arr)
        List.append(list)
    x_train = np.array(List)

    List = []
    for i in range(x_test_path.shape[0]):
        list = []
        for j in range(frames):
            position = [x_test_path[i, j, 0]/8+0.5, 0.5]
            radius = 0.12
            arr = np.zeros((28, 28))
            ro, co = draw.disk((position[0] * 28, position[1] *
                                28), radius=radius*28, shape=arr.shape)
            arr[ro, co] = 1
            list.append(arr)
        List.append(list)
    x_test = np.array(List)

    try:
        np.save(data_path+'SDE_Ball_train_path_{}frames'.format(frames), x_train_path)
        np.save(data_path+'SDE_Ball_test_path_{}frames'.format(frames), x_test_path)
        np.save(data_path+'SDE_Ball_train_{}frames'.format(frames), x_train)
        np.save(data_path+'SDE_Ball_test_{}frames'.format(frames), x_test)
        print('datasets saved for future use')
    except:
        print('could not save datasets')
    print('datasets generated')

x_train = x_train[0:train_size]
x_test = x_test[0:test_size]
x_train_path = x_train_path[0:train_size]
x_test_path = x_test_path[0:test_size]


# Dim: train_size x frames x pictureWidth x pictureHeight x pictureColors
x_train = np.transpose(np.array([x_train]), (1, 2, 3, 4, 0))


# Dim: test_size x frames x pictureWidth x pictureHeight x pictureColors
x_test = np.transpose(np.array([x_test]), (1, 2, 3, 4, 0))


########################################################
# Datensatz für Encoder erstellen

# Dim: train_size, (frames-M+1) x pictureWidth x pictureHeight x (M*pictureColors)
#x_train_longlist = AE_Tools.make_training_data(x_train, train_size, frames, M)


########################################################
# Definitionen

#derivatives = SDE_Tools.make_tensorwise_average_derivatives(M, N, frames, fps)
derivatives = SDE_Tools.make_tensorwise_derivatives(M, frames, fps)

#encoder = AE_Tools.FramewiseEncoder(latent_dim, pictureWidth, pictureHeight, pictureColors, act_CNN, complexity=CNN_complexity, variational=True)
#decoder = AE_Tools.FramewiseDecoder(latent_dim, pictureWidth, pictureHeight, pictureColors, act_CNN, complexity=CNN_complexity)

encoder = AE_Tools.make_Clemens_encoder(latent_dim)
decoder = AE_Tools.make_Clemens_decoder(latent_dim)

ms_Net = SDE_Tools.mu_sig_Net(M, latent_dim, n, act_ms_Net,
                              SDE_Net_complexity, forceHigherOrder=forceHigherOrder)
reconstructor = SDE_Tools.make_Tensorwise_Reconstructor(
    latent_dim*pictureColors, latent_dim*pictureColors, n, Time, frames, ms_Net, batch_size, applyBM=reconstructWithBM)


rec_loss = AE_Tools.make_binary_crossentropy_rec_loss(frames)
ms_rec_loss = SDE_Tools.make_reconstruction_Loss(
    M, n, Time, frames, batch_size, reconstructor, derivatives)
p_loss = SDE_Tools.make_pointwise_Loss(M, latent_dim, Time, frames, ms_Net, batch_size)
cv_loss = SDE_Tools.make_covariance_Loss(latent_dim, Time, frames, batch_size, ms_Net, batch_size)
ss_loss = SDE_Tools.make_sigma_size_Loss(latent_dim, ms_Net)

MSE = tf.keras.losses.MeanSquaredError()


def SDELoss(Z_derivatives, ms_rec):
    S = 0
    S += 2*ms_rec_loss(Z_derivatives, None)
    S += 10*p_loss(Z_derivatives, ms_rec)
    S += 1.6*cv_loss(Z_derivatives, ms_rec)
    # S += 1000*ss_loss(Z_derivatives,ms_rec) #mal ohne probieren
    return S


alpha = 1  # zuletzt 1


def StartingLoss(X_org, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List):
    S = 20*rec_loss(X_org, X_rec_List)
    S += alpha*6*ms_rec_loss(Z_derivatives, Z_rec_List)  # zuletzt 2
    S += alpha*10*p_loss(Z_derivatives, None)
    #S += 0.01/MSE(Z_rec_List, tf.constant(np.zeros(Z_rec_List.shape),dtype=tf.float32))
    #S += 10*MSE(np.ones(Z_enc_List.shape[0]),tf.map_fn(abs,Z_enc_List)/Z_enc_List.shape[1])
    # S += MSE(np.ones(Z_enc_List.shape[0]),tf.map_fn(K.mean,Z_enc_List)) #Betrag vergessen
    # S += alpha*0.05*cv_loss(Z_derivatives, None) #zuletzt ohne
    #S += beta*100*ss_loss(Z_derivatives,None)
    return S


beta = 1


def FullLoss(X_org, Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List):
    S = 20*rec_loss(X_org, X_rec_List)
    S += beta*1*ms_rec_loss(Z_derivatives, Z_rec_List)
    S += beta*10*p_loss(Z_derivatives, None)
    S += beta*1*cv_loss(Z_derivatives, None)
    #S += beta*1000*ss_loss(Z_derivatives, None)
    return S

########################################################
# SDE_VAE definieren


SDE_VAE = SDE_VAE_Tools.SDE_Variational_Autoencoder(
    M, N, encoder, derivatives, reconstructor, decoder, StartingLoss)
# inp hat dim: None x frames x pictureWidth x pictureHeight x pictureColors

print('model defined')


########################################################
# Model ohne SDE-Rekonstruktion die latenten Darstellungen lernen lassen
print('initial training for encoder and decoder to learn a first latent representation')
SDE_VAE.reconstruct_smoothly = False
SDE_VAE.compile(optimizer='adam', loss=lambda x, arg: arg)
SDE_VAE.fit(x_train, x_train, epochs=VAE_epochs_starting, batch_size=batch_size, shuffle=False)
SDE_VAE.summary()

# Latente Darstellung zum testen abspeichern
_, _, Z_enc_List, _, _, _ = SDE_VAE.fullcall(x_train)
np.save(data_path+'TestIfEncoderWorks2', Z_enc_List)


########################################################
# Die SDE-Rekonstruktion der latenten Darstellungen lernen lassen
# Dieses training ist merklich schneller auf der haupt-cpu ohne verwendung einer gpu
print('initial training to learn SDE governing latent representation')
with tf.device('/cpu:0'):
    ms_Net.compile(optimizer='adam', loss=SDELoss, metrics=[
                   ss_loss, lambda x, m: ms_rec_loss(x, None)])
    _, _, Z_enc, _, _, _ = SDE_VAE.fullcall(x_train)
    z_train_derivatives = tf.constant(derivatives(Z_enc), dtype=tf.float32)
    ms_Net.fit(z_train_derivatives, z_train_derivatives,
               epochs=SDE_epochs_starting, batch_size=batch_size, shuffle=False)
    ms_Net.summary()


'''
########################################################
# En-&Decoder und SDE-Rekonstruktion zusammen trainieren
print('main training with SDEs and Decoders combined')
SDE_VAE.custom_loss = FullLoss
SDE_VAE.compile(optimizer='adam', loss=lambda x, arg: arg)
SDE_VAE.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=False)
SDE_VAE.summary()
'''

########################################################
# Modell so einstellen, dass glatter reconstruiert wird.
SDE_VAE.reconstruct_smoothly = True


########################################################
# Ergebnisse speichern
Z_enc_mean_List, Z_enc_log_var_List, Z_enc_List, Z_derivatives, Z_rec_List, X_rec_List = SDE_VAE.fullcall(
    x_test)

np.save(data_path+'Results_SDE_Ball_Z_org_{}frames'.format(frames), x_test_path)
np.save(data_path+'Results_SDE_Ball_X_org_{}frames'.format(frames), x_test)
np.save(data_path+'Results_SDE_Ball_Z_enc_{}frames'.format(frames), Z_enc_List)
np.save(data_path+'Results_SDE_Ball_Z_rec_{}frames'.format(frames), Z_rec_List)
np.save(data_path+'Results_SDE_Ball_X_rec_{}frames'.format(frames), X_rec_List)


########################################################
# Ergebnisse darstellen

#x_test = data.create_dataset(dataset_size=100, frames=10, picture_size=28, object_number=3)
k = 0
print('x_test:', x_test.shape)
_, _, enc_lat, _, rec_lat, rec_imgs = SDE_VAE.fullcall(x_test)
print('rec_imgs:', rec_imgs.shape)
#enc_lat = list(map(lambda i: encoder(x_train[i,:,:,:,:])[-1], range(batch_size)))
#enc_lat = tf.stack(enc_lat, axis=0)

#Z_0 = derivatives(enc_lat)[:,0,:,:]
#rec_lat = reconstructor(Z_0)[:,:,0,:]
#rec_lat = x_test_path

print('enc_lat:', enc_lat.shape)
#rec_lat = reconstructor(enc_lat[:,0,:])
print('rec_lat:', rec_lat.shape)

fig, axs = plt.subplots(9, 10)
for i in range(4):
    for j in range(10):
        axs[2*i, j].imshow(x_test[i, j, :, :, 0], cmap='gray')
        axs[2*i+1, j].imshow(rec_imgs[i, j, :, :, 0], cmap='gray')
for i in range(5):
    axs[8, 2*i].plot(np.linspace(1, frames, frames), enc_lat[i, :, 0], '-')
    axs[8, 2*i+1].plot(np.linspace(1, frames, frames), rec_lat[i, :, 0], '-')
plt.show()
