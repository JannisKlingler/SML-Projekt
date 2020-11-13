# Die gewünschte Visualisierung kopieren und in den Code einfügen.

###############################################################################
##########Plottet n Bilder und die vom VAE erstellten Rekonstruktionen#########
###############################################################################
decoded_imgs = vae.predict(x_train)

n = 15
k = 0
plt.figure(figsize=(20, 4))
for i in np.random.randint(len(x_train), size=n):
    ax = plt.subplot(2, n, k + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, k + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    k = k + 1
plt.show()

###############################################################################
##########Plottet die latenten Darstellungen im zweidimensionalem Raum#########
###############################################################################
n = 4000

encoded_imgs = encoder.predict(x_train[0:n], batch_size=100)
plt.figure(figsize=(8, 8))
plt.scatter(encoded_imgs[2][:, 0], encoded_imgs[2][:, 1], c=y_train[0:n],
            cmap='tab10', marker='o')
plt.colorbar()
plt.show()

###############################################################################
##########Plottet die latenten Darstellungen im dreidimensionalem Raum#########
######Das Schaubild ist interaktiv und kann mit der Maus verändert werden.#####
###############################################################################
n = 3000

encoded_imgs = encoder.predict(x_train[0:n], batch_size=100)
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(encoded_imgs[0][:, 0], encoded_imgs[0][:, 1],
           encoded_imgs[0][:, 2], c=y_train[0:n], cmap='tab10', marker='o')
plt.show()

###############################################################################
######Erstellt eine nxn große Karte des zweidimensionalen lateneten Raums######
###############################################################################
n = 25

figure = np.zeros((28 * n, 28 * n))
for i, yi in enumerate(np.linspace(-2.25, 2.25, n)):
    for j, xi in enumerate(np.linspace(-2.25, 2.25, n)):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.show()
