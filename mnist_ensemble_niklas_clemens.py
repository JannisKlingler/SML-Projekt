import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(0)

# Daten laden und normalisieren
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((len(x_train), 28, 28, 1)) / 255.
x_test = x_test.reshape((len(x_test), 28, 28, 1)) / 255.

epochs = 40
ens_size = 10
models_list = list()
history_list = list()
ens_pred_list = list()

# datagen = ImageDataGenerator(
#   rotation_range=20,
#   width_shift_range=0.05,
#   height_shift_range=0.05,
#   shear_range=0.05,
#   fill_mode='nearest')

# datagen.fit(x_train)
# train_iterator = datagen.flow(x_train, y_train, batch_size=100)


class PredictionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        y_pred = model.predict(x_test)
        pred_list.append(np.argmax(y_pred, axis=1))


# Modell
for i in range(ens_size):
    pred_list = list()
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', metrics=['accuracy'],
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

#    history = model.fit(train_iterator, epochs=epochs,
#                        validation_data=(x_test, y_test), verbose=2)

    print('Training Ensemblemitglied ', i + 1, '/', ens_size, ':')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=100, verbose=2,
                        callbacks=[PredictionCallback()], validation_data=(x_test, y_test))

    history_list.append(history)
    models_list.append(model)
    ens_pred_list.append(pred_list)

ens_acc = np.zeros(epochs)
for j in range(epochs):
    ens_pred_tranposed = np.transpose(np.array([ens_pred_list[i][j] for i in range(ens_size)]))
    ens_pred = [np.argmax(np.bincount(ens_pred_tranposed[i])) for i in range(len(y_test))]
    ens_pred = np.where(y_test - ens_pred != 0, 0, 1)
    ens_acc[j] = sum(ens_pred) / 10000

print(ens_acc)

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
for history in history_list:
    ax1.plot(history.history['accuracy'], color='C0', label='Perfis COPEX')
    ax1.plot(history.history['val_accuracy'], color='#C0C0C0')
ax1.plot(ens_acc, color='C3', linewidth=2.5)
ax1.set_ylim([0.96, 1])
ax1.grid()

ax1.set_title("Modellgenauigkeit")
for history in history_list:
    ax2.plot(history.history['loss'], color='C0')
    ax2.plot(history.history['val_loss'], color='#C0C0C0')
ax2.set_ylim([1.46, 1.53])
ax2.grid()

ax2.set_title("Lossfunktion: kategorische Kreuzentropie")
plt.show()
