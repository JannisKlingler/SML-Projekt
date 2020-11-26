import tensorflow as tf
from keras import backend as K


def Bernoulli_Loss(encoder, decoder):
    μ, log_σ, z = encoder(encoder.inp)
    decoder_output = decoder(z)
    log_p_xz = K.sum(encoder.inp * K.log(decoder_output) +
                     (1. - encoder.inp) * K.log(1. - decoder_output), axis=-1)
    kl_div = .5 * K.sum(1. + 2. * μ - K.square(μ) - 2. * K.exp(log_σ), axis=-1)
    return (- log_p_xz - kl_div)


def BernoulliConv_Loss(encoder, decoder):
    μ, log_σ, z = encoder(encoder.inp)
    decoder_output = decoder(z)
    log_p_xz = 784. * K.mean(tf.keras.losses.binary_crossentropy(encoder.inp, decoder_output))
    kl_div = .5 * K.sum(1. + 2. * log_σ - K.square(μ) - 2. * K.exp(log_σ), axis=-1)
    return (log_p_xz - kl_div)

# To Do:
# https://keras.io/guides/training_with_built_in_methods/ Erklärung wie man die
# Lossfunktion in eine Klasse umschreiben kann, falls wir das mal brauchen sollten.
# Das Problem ist nämlich, dass  Lossfunktionen eigentlich nur zwei Argumente haben
# sollen (Input, Output). Wir brauchen aber vier Argumente (Input, Output, μ, log_σ).

# Unterkapitel: Handling losses and metrics that don't fit the standard signature.
