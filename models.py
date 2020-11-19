import tensorflow as tf
from keras import backend as K


def HelloWorld():
    print('Hello World')


class VAE_Dense_Encoder(tf.keras.Model):
    def __init__(self, encoder_struc, dropout, act):
        l = len(encoder_struc) - 1
        self.inp = x = tf.keras.Input(shape=(encoder_struc[0],))

        for i in range(1, l):
            x = tf.keras.layers.Dense(encoder_struc[i], activation=act)(x)
            if dropout != 0:
                x = tf.keras.layers.Dropout(dropout)(x)

        μ = tf.keras.layers.Dense(encoder_struc[l], name="mu")(x)
        log_σ = tf.keras.layers.Dense(encoder_struc[l], name="log_sig")(x)

        z = tf.keras.layers.Lambda(lambda arg: arg[0] + K.exp(arg[1]) * K.random_normal(
            shape=(K.shape(arg[0])[0], encoder_struc[l]), mean=0.0, stddev=1.0))([μ, log_σ])

        super(VAE_Dense_Encoder, self).__init__(self.inp, [μ, log_σ, z], name="Encoder")
        self.summary()


class Bernoulli_Dense_Decoder(tf.keras.Model):
    def __init__(self, decoder_struc, dropout, act):
        l = len(decoder_struc) - 1
        self.inp = x = tf.keras.Input(shape=(decoder_struc[0],))

        for i in range(1, l):
            x = tf.keras.layers.Dense(decoder_struc[i], activation=act)(x)
            if dropout != 0:
                x = tf.keras.layers.Dropout(dropout)(x)

        outp = tf.keras.layers.Dense(decoder_struc[l], activation="sigmoid")(x)
        super(Bernoulli_Dense_Decoder, self).__init__(self.inp, outp, name="Decoder")
        self.summary()

# To Do:
# 1)
# Implementierung eines Gauß_Dense_Decoder, zum modellieren stetiger Daten.

# Dieser hat zwei Outputlayer, einen Erwartungswertoutput und ein log_var-Output,
# ähnlich wie im Encoder, nur ohne Reparametrisierung. Es muss außerdem eine
# neue Lossfunktion geschrieben werden, da log_p_xz anders berechnet wird.

# 2)
# Implementierung von  VAE_Conv_Encoder, Bernoulli_Conv_Decoder, Gauß_Conv_Decoder
# für Datensätze komplexer als MNIST. Überlegung einer geeigneten Modellstruktur
