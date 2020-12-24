import datasets as data
from keras import backend as K
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

frames = 10
epochs = 60
akt_fun = 'relu'
job = 'rotatingMNIST'  # 'rotatingMNIST' ; 'bouncingBalls'

if job == 'rotatingMNIST':
    latent_dim = 25

    try:
        x_train = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train.npy')
        x_test = np.load('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test.npy')
        x_test_missing = np.load(
            'C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test_missing.npy')

    except:
        print('Dataset is being generated. This may take a few minutes.')
        x_train, x_test, x_test_missing = data.create_dataset_rotatingMNIST(train_dataset_size=60000,
                                                                            test_dataset_size=10000, frames=10, variation=False)

        np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_train', x_train)
        np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test', x_test)
        np.save('C:/Users/Admin/Desktop/Python/Datasets/rotatingMNIST_test_missing', x_test_missing)
        print('Dataset generated')

if job == 'bouncingBalls':
    latent_dim = 30

    try:
        x_train = np.load('C:/Users/Admin/Desktop/Python/Datasets/bouncingBalls.npy')
        x_test = np.load('C:/Users/Admin/Desktop/Python/Datasets/bouncingBalls_test.npy')
        x_test_missing = np.load(
            'C:/Users/Admin/Desktop/Python/Datasets/bouncingBalls_test_missing.npy')

    except:
        print('Dataset is being generated. This may take a few minutes. Ignore possible warnings.')
        x_train = data.create_dataset_bouncingBalls(dataset_size=60000, frames=10,
                                                    picture_size=28, object_number=3, variation=False)
        x_test = data.create_dataset_bouncingBalls(dataset_size=10000, frames=10,
                                                   picture_size=28, object_number=3, variation=False)

        x_test_missing = np.zeros((10000, 28, 28, frames))
        for j in range(len(x_test)):
            for i in range(3):
                x_test_missing[j, :, :, i] = x_test[j, :, :, i]

        np.save('C:/Users/Admin/Desktop/Python/Datasets/bouncingBalls_train', x_train)
        np.save('C:/Users/Admin/Desktop/Python/Datasets/bouncingBalls_test', x_test)
        np.save('C:/Users/Admin/Desktop/Python/Datasets/bouncingBalls_test_missing', x_test_missing)
        print('Dataset generated')


class ODE2_VAE_ConvTime_Encoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):

        self.inp = x = tf.keras.layers.Input(shape=(28, 28, 10))

        first_frame = x[:, :, :, 0]
        first_frame = tf.keras.layers.Reshape((28, 28, 1))(first_frame)
        first_three_frames = x[:, :, :, 0:3]

        pos_enc = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation=act)(first_frame)
        pos_enc = tf.keras.layers.MaxPooling2D((2, 2))(pos_enc)
        pos_enc = tf.keras.layers.Conv2D(32, (3, 3), activation=act)(pos_enc)
        pos_enc = tf.keras.layers.MaxPooling2D((2, 2))(pos_enc)
        pos_enc = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(pos_enc)
        pos_enc = tf.keras.layers.Flatten()(pos_enc)
        s_μ = tf.keras.layers.Dense(latent_dim, activation=act)(pos_enc)
        s_log_σ = tf.keras.layers.Dense(latent_dim, activation=act)(pos_enc)

        s = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([s_μ, s_log_σ])

        vel_enc = tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                                         activation=act)(first_three_frames)
        vel_enc = tf.keras.layers.MaxPooling2D((2, 2))(vel_enc)
        vel_enc = tf.keras.layers.Conv2D(64, (3, 3), activation=act)(vel_enc)
        vel_enc = tf.keras.layers.MaxPooling2D((2, 2))(vel_enc)
        vel_enc = tf.keras.layers.Conv2D(128, (3, 3), activation=act)(vel_enc)
        vel_enc = tf.keras.layers.Flatten()(vel_enc)
        v_μ = tf.keras.layers.Dense(latent_dim, activation=act)(vel_enc)
        v_log_σ = tf.keras.layers.Dense(latent_dim, activation=act)(vel_enc)

        v = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], latent_dim), mean=0.0, stddev=1.0))([v_μ, v_log_σ])

        z = tf.stack([s, v], axis=1)

        sv_μ = tf.keras.layers.Concatenate(axis=1)([s_μ, v_μ])
        sv_σ = K.exp(tf.keras.layers.Concatenate(axis=1)([s_log_σ, v_log_σ]))
        super(ODE2_VAE_ConvTime_Encoder, self).__init__(
            self.inp, [sv_μ, sv_σ, z], name="Encoder")


class Differential_Function(tf.keras.Model):
    def __init__(self, latent_dim, t):

        self.inp = z_t = tf.keras.layers.Input(shape=(2, latent_dim))
        f_s_tv_t = tf.keras.layers.Flatten()(z_t)
        f_s_tv_t = tf.keras.layers.Dense(100, activation='tanh')(f_s_tv_t)
        f_s_tv_t = tf.keras.layers.Dense(100, activation='tanh')(f_s_tv_t)
        f_s_tv_t = tf.keras.layers.Dense(latent_dim, activation='tanh')(f_s_tv_t)

        v_t = z_t[:, 1, :] + t * f_s_tv_t
        s_t = z_t[:, 0, :] + t * z_t[:, 1, :]

        z_t = tf.stack([s_t, v_t], axis=1)

        super(Differential_Function, self).__init__(
            self.inp, [f_s_tv_t, z_t], name="DifferentialFunction")


def latent_trajectory(z_t, steps):
    Lz_t = [z_t]
    Lf_s_tv_t = []
    for i in range(steps):
        f_s_tv_t, z_t = f(z_t)
        Lz_t.append(z_t)
        Lf_s_tv_t.append(f_s_tv_t)

    Lz_t = tf.stack(Lz_t, axis=2)
    Ls_t = Lz_t[:, 0, :, :]
    Lv_t = Lz_t[:, 1, :, :]
    Lf_s_tv_t = tf.stack(Lf_s_tv_t, axis=1)
    return Lf_s_tv_t, Lz_t, Lv_t, Ls_t


class ODE2_Bernoulli_ConvTime_Decoder(tf.keras.Model):
    def __init__(self, frames, latent_dim, act):

        self.inp = Ls_t = tf.keras.layers.Input(shape=(frames, latent_dim))
        reconstr = []
        for i in range(frames):
            dec = Ls_t[:, i, :]
            dec = tf.keras.layers.Dense(64, activation=act)(dec)
            dec = tf.keras.layers.Dense(4 * 4 * 64, activation=act)(dec)
            dec = tf.keras.layers.Reshape((4, 4, 64))(dec)
            dec = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation=act)(dec)
            dec = tf.keras.layers.UpSampling2D(size=(2, 2))(dec)
            dec = tf.keras.layers.Conv2DTranspose(16, (3, 3), activation=act)(dec)
            dec = tf.keras.layers.UpSampling2D(size=(2, 2))(dec)
            dec = tf.keras.layers.Conv2DTranspose(
                1, (3, 3), padding="same", activation='sigmoid')(dec)
            reconstr.append(dec)
        outp = tf.keras.layers.Concatenate(axis=-1)(reconstr)

        super(ODE2_Bernoulli_ConvTime_Decoder, self).__init__(self.inp, outp, name="Decoder")


def Reconstruction_Loss(x, x_rec):
    return K.sum(frames * tf.keras.losses.binary_crossentropy(x, x_rec), axis=(1, 2))


# Helfer zur Berechnung des Bernoulli_ODE2_Loss
def encode_whole_sequence(x):
    Lsv_μ, Lsv_σ, Lz = [], [], []
    for i in range(frames):
        if job == 'rotatingMNIST':
            if (i < frames-2):
                three_frames = x[:, :, :, i:i+3]
            elif (i >= frames-2):
                three_frames = tf.concat([x[:, :, :, i:frames], x[:, :, :, 0:i-frames+3]], axis=3)
        else:
            if (i < frames-2):
                three_frames = x[:, :, :, i:i+3]
            elif (i >= frames-2):
                three_frames = x[:, :, :, frames-3:frames]
        c = tf.concat([three_frames, x[:, :, :, 0:7]], axis=3)
        sv_μ, sv_σ, z = encoder(c)
        Lsv_μ.append(sv_μ)
        Lsv_σ.append(sv_σ)
        Lz.append(z)

    Lsv_μ = tf.stack(Lsv_μ, axis=1)
    Lsv_σ = tf.stack(Lsv_σ, axis=1)
    Lz = tf.stack(Lz, axis=1)
    z_0 = Lz[:, 0, :, :]
    return Lsv_μ, Lsv_σ, Lz, z_0


t = 1e-5
A = []
for i in range(latent_dim):
    a = np.zeros((2, latent_dim))
    a[1, i] = t
    A.append(a)
A = tf.constant(np.array(A).astype('Float32'))


def ODE2_Flow(Lf_s_tv_t, Lz_t, j, t):
    delta = [f(Lz_t[:, :, j, :] + A[i])[0] - Lf_s_tv_t[:, j, :] for i in range(latent_dim)]
    delta = tf.stack(delta, axis=1)

    Trace = tf.linalg.trace(delta) / tf.constant(t)
    return Trace


γ = 1.


def Bernoulli_ODE2_Loss(x, x_pred):
    Lsv_μ, Lsv_σ, Lz, z_0 = encode_whole_sequence(x)
    Lf_s_tv_t, Lz_t, _, Ls_t = latent_trajectory(z_0, steps=frames-1)
    x_rec = decoder(Ls_t)

    log_p_x_z = - Reconstruction_Loss(x, x_rec)

    log_pz = tfd.MultivariateNormalDiag(loc=tf.zeros(
        2*frames*latent_dim), scale_diag=tf.ones(2*frames*latent_dim)).log_prob(tf.keras.layers.Flatten()(Lz_t))

    log_qz_0 = tfd.MultivariateNormalDiag(loc=(Lsv_μ[:, 0, :]), scale_diag=(
        Lsv_σ[:, 0, :])).log_prob(tf.keras.layers.Flatten()(z_0))

    Traces = [ODE2_Flow(Lf_s_tv_t, Lz_t, j, t=1e-5) for j in range(frames - 1)]
    Traces = tf.stack(Traces, axis=1)
    Int = K.cumsum(Traces, axis=1)

    log_qz = frames * log_qz_0 - K.sum(Int, axis=1)

    ELBO = log_p_x_z + log_pz - log_qz

    if γ > 0:
        log_q_enc = tfd.MultivariateNormalDiag(loc=(tf.keras.layers.Flatten()(Lsv_μ)), scale_diag=(
            tf.keras.layers.Flatten()(Lsv_σ))).log_prob(tf.keras.layers.Flatten()(Lz))

        ELBO -= γ * (log_qz - log_q_enc)

    return - ELBO


encoder = ODE2_VAE_ConvTime_Encoder(frames, latent_dim, akt_fun)
f = Differential_Function(latent_dim, t=1e-5)
decoder = ODE2_Bernoulli_ConvTime_Decoder(frames, latent_dim, akt_fun)
ode2vae = tf.keras.Model(encoder.inp, decoder(
    latent_trajectory(encoder(encoder.inp)[-1], steps=frames-1)[-1]))


# Verfügbare Lossfunktionen: Bernoulli_ODE2_Loss, Reconstruction_Loss
ode2vae.compile(optimizer='adam', loss=Bernoulli_ODE2_Loss,
                metrics=[tf.keras.metrics.MeanSquaredError(name="MSE"), Reconstruction_Loss])

ode2vae.fit(x_train, x_train, epochs=epochs, batch_size=100,
            validation_data=(x_test[0:500], x_test[0:500]))


k = 0
rec_imgs = ode2vae.predict(x_test)
np.save('C:/Users/Admin/Desktop/Python/Datasets/reconstructions', rec_imgs)
fig, index = plt.figure(figsize=(10, 10)), np.random.randint(len(x_test), size=5)
grid = ImageGrid(fig, 111,  nrows_ncols=(10, 10), axes_pad=0.1,)
plot = [x_test_missing[index[0]][:, :, j] for j in range(10)]
for i in index:
    if k != 0:
        original = [x_test_missing[i][:, :, j] for j in range(10)]
        plot = np.vstack((plot, original))
    reconst = [rec_imgs[i][:, :, j] for j in range(10)]
    plot = np.vstack((plot, reconst))
    k += 1
for ax, im in zip(grid, plot):
    plt.gray()
    ax.imshow(im)
plt.show()


#    ode_regul = 0.01 * np.sum([np.sum(f.get_weights()[i] ** 2)
#                               for i in range(len(f.get_weights()))])
