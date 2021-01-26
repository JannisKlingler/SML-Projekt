import datasets as data
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# %% hyperparameter
epochs = 15
latent_dim = 25  # Dimensionality for latent variables. 20 works fine.
batch_size = 100  # ≥100 as suggested by Kingma in Autoencoding Variational Bayes.
train_size = 60000  # Data points in train set. Choose accordingly to dataset size.
test_size = 10000  # Data points in test set. Choose accordingly to dataset size.
batches = int(train_size / batch_size)
frames = 10  # Number of images in every datapoint. Choose accordingly to dataset size.
armortized_len = 4  # Sequence size seen by velocity encoder network. Needs to be ≥3
act = 'relu'  # Activation function 'tanh' is used in odenet.

ode_integration = 'trivialsum'  # options: 'DormandPrince' , 'trivialsum'
# we suggest 'trivialsum' as it is very fast and yields good results.
eval_interval = 20  # Time between evaluation on test batch during training.
data_path = 'C:/Users/Admin/Desktop/Python/Datasets/'
job = 'bouncingBalls'  # Dataset for training. Options: 'rotatingMNIST' , 'bouncingBalls'

# %%
try:
    x_train = np.load(data_path + job + '_train.npy')
    x_test = np.load(data_path + job + '_test.npy')

except:
    print('Dataset is being generated. This may take a few minutes.')

    if job == 'rotatingMNIST':
        x_train, x_test, _ = data.create_dataset_rotatingMNIST(
            train_dataset_size=train_size, test_dataset_size=test_size, frames=frames, variation=False)

    if job == 'bouncingBalls':
        x_train = data.create_dataset_bouncingBalls(dataset_size=train_size, frames=frames,
                                                    picture_size=28, object_number=3, variation=False, pictures=True)
        x_test = data.create_dataset_bouncingBalls(dataset_size=test_size, frames=frames,
                                                   picture_size=28, object_number=3, variation=False, pictures=True)

    np.save(data_path + job + '_train', x_train)
    np.save(data_path + job + '_test', x_test)
    print('Dataset generated')

x_train = x_train[:train_size]  # (train_size, 28, 28, frames)
x_test = x_test[:test_size]  # (train_size, 28, 28, frames)

train_dataset = (tf.data.Dataset.from_tensor_slices(x_train)
                 .shuffle(train_size).batch(batch_size))  # (batch_size, 28, 28, frames)
test_dataset = (tf.data.Dataset.from_tensor_slices(x_test)
                .shuffle(test_size).batch(batch_size))  # (batch_size, 28, 28, frames)


class ODE2VAE(tf.keras.Model):
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
            tf.keras.layers.Dense(latent_dim + latent_dim)])

        self.velocity_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((28, 28, armortized_len)),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=act),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation=act),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation=act),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=act),
            tf.keras.layers.Dense(latent_dim + latent_dim)])

        self.differential_equation = tf.keras.Sequential([
            tf.keras.layers.InputLayer((2, latent_dim)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4 * latent_dim, activation='tanh'),
            tf.keras.layers.Dense(4 * latent_dim, activation='tanh'),
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
    def encode(self, x, i):
        pos_mean, pos_logsig = tf.split(self.position_encoder(
            x[:, :, :, i]), num_or_size_splits=2, axis=1)
        vel_mean, vel_logsig = tf.split(self.velocity_encoder(
            x[:, :, :, i:i + armortized_len]), num_or_size_splits=2, axis=1)
        return tf.stack([pos_mean, vel_mean, pos_logsig, vel_logsig], axis=1)

    def reparameterize(self, x_encoded):
        pos_eps = tf.random.normal(shape=(batch_size, latent_dim))
        vel_eps = tf.random.normal(shape=(batch_size, latent_dim))
        pos = pos_eps * tf.exp(x_encoded[:, 2]) + x_encoded[:, 0]
        vel = vel_eps * tf.exp(x_encoded[:, 3]) + x_encoded[:, 1]
        return tf.stack([pos, vel], axis=1)

    def ode_system(self, t, z, epsilon):

        if ode_integration == 'trivialsum':
            d_vel_dt = self.differential_equation(z)
            d_pos_dt = z[:, 1] + d_vel_dt
            return tf.stack([d_pos_dt, d_vel_dt], axis=1)

        elif ode_integration == 'DormandPrince':
            s_t, v_t = z[0][:, 0], z[0][:, 1]
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(v_t)
                d_vel_dt = self.differential_equation(tf.stack([s_t, v_t], axis=1))
                d_pos_dt = v_t
                g = tf.reduce_sum(epsilon * d_vel_dt, axis=1)
            trace = tf.reduce_sum(epsilon * tape.gradient(g, v_t), axis=1)
            return tf.stack([d_pos_dt, d_vel_dt], axis=1), - trace

    def latent_trajectory(self, z_t, log_qz_t, ode_integration):
        latent_states_ode = [z_t]
        log_qz_ode = [log_qz_t]

        if ode_integration == 'DormandPrince':
            epsilon = tf.random.normal(shape=(batch_size, latent_dim))
            t = tf.linspace(0, frames-1, num=frames)
            latent_trajectory = tfp.math.ode.DormandPrince().solve(
                self.ode_system, 0, (z_t, log_qz_t), t, constants={'epsilon': epsilon})
            latent_states_ode = latent_trajectory.states[0]
            log_qz_ode = latent_trajectory.states[1]

        elif ode_integration == 'trivialsum':
            for i in range(frames - 1):
                z_t += self.ode_system(1, z_t, [])
                latent_states_ode.append(z_t)
                log_qz_ode.append(log_qz_t)

        return tf.stack(latent_states_ode), tf.stack(log_qz_ode)

    def decode(self, pos):
        x_rec = self.decoder(pos)
        return x_rec[:, :, :, 0]


model = ODE2VAE(latent_dim, act, armortized_len)
optimizer = tf.keras.optimizers.Adam()


def compute_loss(model, x, ode_integration):

    if ode_integration == 'trivialsum':
        x_encoded = model.encode(x, 0)
        z_0 = model.reparameterize(x_encoded)
        latent_states_ode, _ = model.latent_trajectory(z_0, [], ode_integration)
        x_rec = tf.stack([model.decode(latent_states_ode[i, :, 0]) for i in range(frames)], axis=3)
        return tf.reduce_sum(frames * tf.keras.losses.binary_crossentropy(x, x_rec))

    elif ode_integration == 'DormandPrince':
        x_encoded = tf.stack([model.encode(x, i) for i in range(frames - armortized_len + 1)])
        latent_states_enc = tf.stack([model.reparameterize(x_encoded[i])
                                      for i in range(frames - armortized_len + 1)])

        log_qz_enc = tf.reduce_sum(tfd.MultivariateNormalDiag(
            loc=x_encoded[:, :, :2], scale_diag=2 * tf.math.exp(x_encoded[:, :, 2:])).log_prob(latent_states_enc), axis=2)

        latent_states_ode, log_qz_ode = model.latent_trajectory(
            latent_states_enc[0], log_qz_enc[0], ode_integration)

        x_rec = tf.stack([model.decode(latent_states_ode[i, :, 0]) for i in range(frames)], axis=3)

        log_pz_normal = tf.reduce_sum(tfd.MultivariateNormalDiag(
            loc=tf.zeros(shape=(frames-1, batch_size, 2, latent_dim)),
            scale_diag=tf.ones(shape=(frames-1, batch_size, 2, latent_dim))).log_prob(latent_states_ode[1:]))

        log_p_x_z = - tf.reduce_sum(frames * tf.keras.losses.binary_crossentropy(x, x_rec))

        log_qz_ode = tf.reduce_sum(log_qz_ode[1:frames - armortized_len])

        KL_qz_ode_qz_enc = log_qz_ode - tf.reduce_sum(log_qz_enc[1:])

        KL_qz_ode_pz_normal = log_qz_ode - log_pz_normal

        KL_qz_0_enc_pz_0 = - 0.5 * tf.reduce_sum(2. * x_encoded[:, :, 1:] - tf.math.square(x_encoded[:, :, :1])
                                                 - 2. * tf.math.exp(x_encoded[:, :, 1:]))

        ELBO = log_p_x_z - KL_qz_0_enc_pz_0 - KL_qz_ode_qz_enc - KL_qz_ode_pz_normal

        return - ELBO


def evaluate_during_training(model, test_sample, ode_integration, time_history, progress_ep, progress_bat, steps):
    x_encoded = model.encode(test_sample, 0)
    z_0 = model.reparameterize(x_encoded)
    latent_states_ode, _ = model.latent_trajectory(z_0, tf.zeros(batch_size), ode_integration)
    x_rec = np.array(tf.stack([model.decode(latent_states_ode[i, :, 0])
                               for i in range(frames)], axis=3))

    mean = tf.keras.metrics.Mean()
    mean(compute_loss(model, test_sample, ode_integration))
    elbo = - mean.result() / batch_size
    mean(tf.reduce_sum(frames * tf.keras.losses.binary_crossentropy(test_sample, x_rec)))
    rec = - mean.result() / batch_size

    meansqerr = tf.keras.metrics.MeanSquaredError()
    meansqerr(test_sample, x_rec)
    mse = meansqerr.result()

    if progress_bat == 0:
        return int(elbo), int(rec), mse, '-', '--'
    elif progress_bat < 10:
        a = time_history[progress_ep, progress_bat]
        b = time_history[progress_ep, 0]
        min, sec = divmod((a-b)/(progress_bat)*(steps-progress_bat-1), 60)
        return int(elbo), int(rec), mse, int(min), int(sec)
    elif progress_bat != steps - 1:
        a = time_history[progress_ep, progress_bat]
        b = time_history[progress_ep, progress_bat-10]
        min, sec = divmod((a-b)/10*(steps-progress_bat-1), 60)
        return int(elbo), int(rec), mse, int(min), int(sec)
    else:
        fig = plt.figure(figsize=(8, 8))
        index = np.random.randint(batch_size, size=5)
        grid = ImageGrid(fig, 111,  nrows_ncols=(2*5, frames), axes_pad=0.1,)
        plot = np.zeros((2*5*frames, 28, 28))
        for j, i in enumerate(index):
            plot[j*2*frames:(j*2+1)*frames] = np.transpose(np.array(test_sample[i]), (2, 0, 1))
            plot[(j*2+1)*frames:(j*2+2)*frames] = np.transpose(x_rec[i], (2, 0, 1))
        for ax, im in zip(grid, plot):
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow(im)
        plt.show()

        a = time_history[progress_ep, progress_bat-1]
        b = time_history[progress_ep, 0]
        min, sec = divmod((a-b)*(epochs-progress_ep-1), 60)
        min_el, sec_el = divmod((a-b), 60)
        return int(min), int(sec), int(min_el), int(sec_el)


for test_batch in test_dataset.take(1):
    test_sample = test_batch[:batch_size]

time_history = np.zeros((epochs, batches))
evaluate_during_training(
    model, test_sample, ode_integration, time_history, 0, batches-1, batches)


@ tf.function
def train_step(model, x, optimizer, ode_integration):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, ode_integration)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


print('Compiling Model. This may take up to a few minutes.')
print('Model will be evaluated on a random test batch every {}s.'.format(eval_interval))


for epoch in range(1, epochs + 1):
    print_time = time.time()
    for batch, train_x in enumerate(train_dataset):
        train_step(model, train_x, optimizer, ode_integration)
        time_history[epoch - 1, batch] = time.time()
        if time_history[epoch - 1, batch] - print_time > eval_interval and batch + 1 != batches:
            print_time = time.time()
            elbo, rec, mse, min, sec = evaluate_during_training(
                model, test_sample, ode_integration, time_history, epoch - 1, int(batch), batches)
            print('Batch: {}/{} | Epoch: {}/{}'.format(str(batch+1).zfill(3),
                                                       batches, epoch, epochs), end='')
            if ode_integration == 'trivialsum':
                print(' | ETA: {}:{} | Metrics on random test batch: Reconstruction Loss {}, MSE per pixel {:.5f}'.format(
                    min, str(sec).zfill(2), rec, mse))
            if ode_integration == 'DormandPrince':
                print(' | ETA: {}:{} | Metrics on random test batch: ELBO {}, Reconstruction Loss {}, MSE per pixel {:.5f}'.format(
                    min, str(sec).zfill(2), elbo, rec, mse))
    min, sec, min_el, sec_el = evaluate_during_training(
        model, test_sample, ode_integration, time_history, epoch - 1, int(batch), batches)
    print('Epoch {}/{} completed. Time elapsed this epoch: {}:{}'.format(epoch,
                                                                         epochs, min_el, str(sec_el).zfill(2)), end='')
    print(' Estimated time until completion of training: {}:{}.'.format(min, str(sec).zfill(2)))
