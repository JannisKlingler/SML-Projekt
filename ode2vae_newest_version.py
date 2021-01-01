import time
import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
# import datasets as data


# Wird bei mir für GPU-Unterstüzung benötigt.
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# %% Hyperparameter
epochs = 4
latent_dim = 20
batch_size = 100
frames = 10
armortized_len = 3
train_size = 60000  # max. 60000
test_size = 10000  # max. 10000
act = 'relu'
ode_type = 'discrete'  # 'continous' # discrete train time ~200xfaster'
data_path = 'C:/Users/Admin/Desktop/Python/Datasets/'
job = 'bouncingBalls'  # 'rotatingMNIST' ; 'bouncingBalls'

# %%

try:
    x_train = np.load(data_path + job + '_train.npy')
    x_test = np.load(data_path + job + '_test.npy')

except:
    print('Dataset is being generated. This may take a few minutes.')

    if job == 'rotatingMNIST':
        x_train, x_test, _ = data.create_dataset_rotatingMNIST(
            train_dataset_size=60000, test_dataset_size=10000, frames=10, variation=False)

    if job == 'bouncingBalls':
        x_train = data.create_dataset_bouncingBalls(dataset_size=60000, frames=10,
                                                    picture_size=28, object_number=3, variation=False, pictures=True)
        x_test = data.create_dataset_bouncingBalls(dataset_size=10000, frames=10,
                                                   picture_size=28, object_number=3, variation=False, pictures=True)

    np.save(data_path + job + '_train', x_train)
    np.save(data_path + job + '_test', x_test)
    print('Dataset generated')

x_train = x_train[0:train_size]
x_test = x_test[0:test_size]

train_dataset = (tf.data.Dataset.from_tensor_slices(x_train)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(x_test)
                .shuffle(test_size).batch(batch_size))


class ODE2VAE(tf.keras.Model):
    """Convolutional Ordinary Differential Equation Variational Autoencoder."""

    def __init__(self, latent_dim, act, armortized_len):
        super(ODE2VAE, self).__init__()
        self.latent_dim = latent_dim
        self.act = act
        self.armortized_len = armortized_len

        self.position_encoder = tf.keras.Sequential([
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.InputLayer((28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=act),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation=act),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation=act),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=act),
            tf.keras.layers.Dense(latent_dim + latent_dim, activation=act)])

        self.velocity_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((28, 28, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=act),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation=act),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation=act),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=act),
            tf.keras.layers.Dense(latent_dim + latent_dim, activation=act)])

        self.differential_equation = tf.keras.Sequential([
            tf.keras.layers.InputLayer((2, latent_dim)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='tanh'),  # ??
            tf.keras.layers.Dense(100, activation='tanh'),  # ??
            tf.keras.layers.Dense(latent_dim)])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((latent_dim,)),
            tf.keras.layers.Dense(256, activation=act),
            tf.keras.layers.Dense(4 * 4 * 128, activation=act),
            tf.keras.layers.Reshape((4, 4, 128)),
            tf.keras.layers.Conv2DTranspose(64, (3, 3), activation=act),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2DTranspose(32, (3, 3), activation=act),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2DTranspose(1, (3, 3), padding="same", activation='sigmoid')])

    @tf.function
    def encode(self, x):
        pos_mean, pos_logsig = tf.split(self.position_encoder(
            x[:, :, :, 0]), num_or_size_splits=2, axis=1)
        vel_mean, vel_logsig = tf.split(self.velocity_encoder(
            x[:, :, :, 0:armortized_len]), num_or_size_splits=2, axis=1)
        return pos_mean, pos_logsig, vel_mean, vel_logsig

    def reparameterize(self, pos_mean, pos_logsig, vel_mean, vel_logsig):
        pos_eps = tf.random.normal(shape=pos_mean.shape)
        vel_eps = tf.random.normal(shape=vel_mean.shape)
        pos = pos_eps * tf.exp(pos_logsig) + pos_mean
        vel = vel_eps * tf.exp(vel_logsig) + vel_mean
        return pos, vel

    def ode_system(self, t, z):
        if ode_type == 'continous':
            s_t = z[0][0, :, :]
            v_t = z[0][1, :, :]
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(v_t)
                d_vel_dt = self.differential_equation(tf.stack([s_t, v_t], axis=1))
                d_pos_dt = v_t + 0.05 * d_vel_dt  # ?
                d_vel_di = tf.transpose([tape.gradient(d_vel_dt[:, i], v_t)
                                         for i in range(latent_dim)], perm=(1, 0, 2))
                d_log_q_dt = - tf.linalg.trace(d_vel_di)
            return tf.stack([d_pos_dt, d_vel_dt]), d_log_q_dt
        if ode_type == 'discrete':
            d_vel_dt = self.differential_equation(tf.transpose(z, (1, 0, 2)))
            d_pos_dt = z[1, :, :]  # + dt * d_vel_dt?
            return tf.stack([d_pos_dt, d_vel_dt])

    def latent_trajectory(self, z_t, log_qz_t, ode_type):
        eval_points = []
        Llog_qz_t = []
        if ode_type == 'continous':
            dt = 0.05
            t = tf.linspace(dt, 1, num=int(np.floor(1/dt-1)))
            for i in range(frames-1):
                latent_trajectory = tfp.math.ode.DormandPrince().solve(self.ode_system, 0, (z_t, log_qz_t), t)
                z_t = latent_trajectory.states[0][int(np.floor(1/dt-2))]
                log_qz_t = latent_trajectory.states[1][int(np.floor(1/dt-2))]
                eval_points.append(z_t)
                Llog_qz_t.append(log_qz_t)

        if ode_type == 'discrete':
            for i in range(frames-1):
                z_t += self.ode_system(1, z_t)
                eval_points.append(z_t)

        return eval_points, Llog_qz_t

    def decode(self, pos):
        x_rec = self.decoder(pos)
        return x_rec


model = ODE2VAE(latent_dim, act, armortized_len)
optimizer = tf.keras.optimizers.Adam()


def compute_loss(model, x, ode_type):
    pos_mean, pos_logsig, vel_mean, vel_logsig = model.encode(x)
    pos, vel = model.reparameterize(pos_mean, pos_logsig, vel_mean, vel_logsig)
    z_0 = tf.stack([pos, vel])

    log_qz_0 = tf.reduce_sum(tfd.MultivariateNormalDiag(loc=tf.stack([pos_mean, vel_mean]),
                                                        scale_diag=tf.stack(2 * tf.math.exp([pos_logsig, pos_logsig]))).log_prob(z_0), axis=0)

    eval_points, Llog_qz_t = model.latent_trajectory(z_0, log_qz_0, ode_type)

    Lx_rec = []
    for i in range(frames):
        if i == 0:
            x_rec = model.decode(z_0[0, :, :])
        else:
            x_rec = model.decode(eval_points[i-1][0])
        x_rec = tf.reshape(x_rec, shape=(tf.shape(x_rec)[0], 28, 28))
        Lx_rec.append(x_rec)

    Lx_rec = tf.stack(Lx_rec, axis=-1)

    log_p_x_z = - tf.reduce_sum(frames * tf.keras.losses.binary_crossentropy(x, Lx_rec))

    if ode_type == 'discrete':
        return - log_p_x_z

    if ode_type == 'continous':

        log_pz = tf.reduce_sum(tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim),
                                                          scale_diag=tf.ones(latent_dim)).log_prob(tf.concat([eval_points, [z_0]], axis=0)))

        log_qz = tf.reduce_sum(log_qz_0) + tf.reduce_sum(Llog_qz_t)

        return - log_p_x_z - log_pz + log_qz


def reconstruct_random_images(model, test_sample, ode_type):
    images = np.random.randint(tf.shape(test_sample)[0], size=2)
    pos_mean, pos_logsig, vel_mean, vel_logsig = model.encode(test_sample)
    pos, vel = model.reparameterize(pos_mean, pos_logsig, vel_mean, vel_logsig)
    z_0 = tf.stack([pos, vel])

    log_qz_0 = tf.reduce_sum(tfd.MultivariateNormalDiag(loc=tf.stack([pos_mean, vel_mean]),
                                                        scale_diag=tf.stack(2 * tf.math.exp([pos_logsig, pos_logsig]))).log_prob(z_0), axis=0)

    eval_points, _ = model.latent_trajectory(z_0, log_qz_0, ode_type)

    Lx_rec = []
    for i in range(frames):
        if i == 0:
            x_rec = model.decode(z_0[0, :, :])
        else:
            x_rec = model.decode(eval_points[i-1][0])
        x_rec = tf.reshape(x_rec, (tf.shape(x_rec)[0], 28, 28))
        Lx_rec.append(x_rec)

    Lx_rec = tf.stack(Lx_rec, axis=-1)

    mean = tf.keras.metrics.Mean()
    mean(compute_loss(model, test_sample, ode_type))
    elbo = - mean.result() / batch_size
    print('ELBO per sequence on test batch: {}'.format(elbo))

    test_sample = np.array(test_sample)
    Lx_rec = np.array(Lx_rec)

    plt.figure(figsize=(10, 4))
    for i in range(frames):
        for j in range(2):
            ax = plt.subplot(4, frames, i + 1 + (2*j)*frames)
            plt.imshow(test_sample[images[j], :, :, i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(4, frames, i + 1 + (2*j+1)*frames)
            plt.imshow(Lx_rec[images[j], :, :, i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:batch_size, :, :, :]

reconstruct_random_images(model, test_sample, ode_type)


@ tf.function
def train_step(model, x, optimizer, ode_type):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, ode_type)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


train_time = time.time()
for epoch in range(1, epochs + 1):
    batch_counter = 0
    print('Training started')
    for train_x in train_dataset:
        train_step(model, train_x, optimizer, ode_type)
        if batch_counter == 0:
            epoch_time = time.time()
            batch_time = time.time()
        end_time_batch = time.time()
        batch_counter += 1
        if end_time_batch - batch_time > 10 and batch_counter > 1:
            batch_time = time.time()
            eta_min, eta_sec = divmod(int((batch_time - epoch_time) *
                                          (train_size / batch_size) / (batch_counter - 1) - (batch_time - epoch_time)), 60)
            print('Batch {}/{} of Epoch {}/{} completed.'.format(batch_counter,
                                                                 int(train_size/batch_size), epoch, epochs), end='')
            print(' Estimated time until completion of epoch: {} min {} s.'.format(eta_min, eta_sec))

    end_time_epoch = time.time()
    eta_min, eta_sec = divmod(int((end_time_epoch-train_time) *
                                  epochs / epoch - (end_time_epoch-train_time)), 60)
    min, sec = divmod(int(end_time_epoch-epoch_time), 60)
    print('Epoch {}/{} completed. Time elapsed this epoch: {} min {} s'.format(epoch, epochs, min, sec,), end='')
    print(' Estimated time until completion of training: {} min {} s.'.format(eta_min, eta_sec))

    reconstruct_random_images(model, test_sample, ode_type)
