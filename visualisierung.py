# Die gewünschte Visualisierung kopieren und in den Code einfügen.

###############################################################################
##########Plottet n Bilder und die vom VAE erstellten Rekonstruktionen#########
###############################################################################
from matplotlib import animation
rec_imgs = vae.predict(x_train)

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
    plt.imshow(rec_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    k = k + 1
plt.show()

###############################################################################
##########Plottet die latenten Darstellungen im zweidimensionalem Raum#########
###############################################################################

if latent_dim == 2:
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

if latent_dim == 3:
    n = 3000
    encoded_imgs = encoder.predict(x_train[0:n], batch_size=100)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_imgs[2][:, 0], encoded_imgs[2][:, 1],
               encoded_imgs[2][:, 2], c=y_train[0:n], cmap='tab10', marker='o')
    plt.show()

###############################################################################
######Erstellt eine nxn große Karte des zweidimensionalen lateneten Raums######
###############################################################################

if latent_dim == 2:
    n = 25
    encoded_imgs = encoder.predict(x_train)
    x_1 = np.min(encoded_imgs[2][:, 0]) - np.min(encoded_imgs[2][:, 0]) / 4
    x_2 = np.max(encoded_imgs[2][:, 0]) - np.max(encoded_imgs[2][:, 0]) / 4
    y_1 = np.min(encoded_imgs[2][:, 1]) - np.min(encoded_imgs[2][:, 1]) / 4

    figure = np.zeros((28 * n, 28 * n))
    for i, yi in enumerate(np.linspace(x_1, x_2, n)):
        for j, xi in enumerate(np.linspace(y_1, y_1 + abs(x_1 - x_2), n)):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(28, 28)
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.show()

###############################################################################
#Erstellt eine Animation des dreidimensionalen lateneten Raums (zeitaufwändig)#
###############################################################################

if latent_dim == 3:
    n = 25
    frames = 60
    encoded_imgs = encoder.predict(x_train)
    x_1 = np.min(encoded_imgs[2][:, 0]) - np.min(encoded_imgs[2][:, 0]) / 4
    x_2 = np.max(encoded_imgs[2][:, 0]) - np.max(encoded_imgs[2][:, 0]) / 4
    y_1 = np.min(encoded_imgs[2][:, 1]) - np.min(encoded_imgs[2][:, 1]) / 4
    z_1 = np.min(encoded_imgs[2][:, 2]) - np.min(encoded_imgs[2][:, 2]) / 4

    fig = plt.figure(figsize=(10, 10))
    ims = []
    for k in range(frames):
        figure = np.zeros((28 * n, 28 * n))
        for i, yi in enumerate(np.linspace(x_1, x_2, n)):
            for j, xi in enumerate(np.linspace(y_1, y_1 + abs(x_1 - x_2), n)):
                z_sample = np.array([[xi, yi, z_1 + k * abs(x_1 - x_2) / frames]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(28, 28)
                figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
        print("Frame", k+1, " von ", frames)
        im = plt.imshow(figure, cmap='gray', animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save('latenter_raum.gif', writer='imagemagick', fps=30)
    plt.show()
