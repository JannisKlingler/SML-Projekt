import tensorflow as tf
from keras import backend as K


def HelloWorld():
    print('Hello World')

class VAE_Dense_Encoder(tf.keras.Model):
    def __init__(self, data_dim, latent_dim, hidden_sizes, activation_function_name):
        self.inp = tf.keras.Input(shape=(data_dim,))
        x = tf.keras.layers.Dense(500, activation="tanh")(self.inp)
        #x = tf.keras.layers.Lambda(lambda y:y)
        #for L in hidden_sizes:
        #    x = tf.keras.layers.Dense(L, activation=activation_function_name)(x)
        #x = x(self.inp)
        mu = tf.keras.layers.Dense(latent_dim, name="mu")(x)
        sig = tf.keras.layers.Dense(latent_dim, name="log_sig")(x)
        z = tf.keras.layers.Lambda(lambda a : a[0] + K.exp(a[1]) * K.random_normal(shape=(K.shape(a[0])[0], latent_dim), mean=0., stddev=1))([mu, sig])
        super(VAE_Dense_Encoder, self).__init__(self.inp, [mu, sig, z], name="Encoder")
        self.summary()

class Bernoulli_Dense_Decoder(tf.keras.Model):
    def __init__(self, data_dim, latent_dim, hidden_sizes, activation_function_name):
        self.inp = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(500, activation="tanh")(self.inp)
        x = tf.keras.layers.Dense(data_dim, activation="sigmoid")(x)
        super(Bernoulli_Dense_Decoder, self).__init__(self.inp, x, name="Decoder")
        self.summary()
