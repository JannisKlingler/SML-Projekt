import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from mpl_toolkits.axes_grid1 import ImageGrid


class Balls:
    position = [np.array([]), np.array([])]
    radius = np.array([])
    velocity = [np.array([]), np.array([])]

    def __init__(self, width, height, amount=1):
        max_radius = 4 # Maximaler Radius eines Balles
        max_velocity = 4 # Maximale Geschwindigkeit
        self.radius = np.random.rand(amount) * max_radius

        x_pos = max_radius + np.random.rand(amount) * max((width - 2 * max_radius), 0)
        y_pos = max_radius + np.random.rand(amount) * max((height - 2 * max_radius), 0)
        self.position = [x_pos, y_pos]

        self.velocity = np.random.rand(amount) * max_velocity

    def pixel_in_ball(self, pos_pixel):
        # pos_pixel = [x, y]
        # returns whether a pixel is inside of some ball

        for i in range(len(self.radius)):
            if math.pow((pos_pixel[0] - self.position[0][i]), 2) + math.pow((pos_pixel[1] - self.position[1][i]), 2) < radius[i]:
               return 0.0

        return 1.0



def create_picture(balls, width=28, height=28):

    #background
    pixel = 1.0
    pic = np.tile(pixel, (height, width))

    for x in range(len(pic)):
        for y in range(len(pic[0])):
            pic[x][y] = balls.pixel_in_ball([x, y])

    return pic

def create_animation_images(width = 28, height = 28, frames = 10):
    vel1 = np.array([random.random() - 0.5 , random.random() - 0.5])
    vel2 = np.array([random.random() - 0.5, random.random() - 0.5])

    left = np.array([width/3.0, height/3.0])
    right = np.array([2 * width/3.0, 2 * height/3.0])

    images = []
    for i in range(frames):
        left += vel1
        right += vel2
        rect = np.array([left, right])

        images.append(create_picture(width=width, height=height, rect=rect))
        #images.append([create_picture(width=width,
        #                             height=height,
        #                             rect=rect,
        #                             animated=True)])
    return np.transpose(np.array(images), [1, 2, 0])


def show_dataset(x_test, rec_imgs):
    k = 0
    fig, index = plt.figure(figsize=(10, 10)), np.random.randint(len(x_test), size=5)
    grid = ImageGrid(fig, 111,  nrows_ncols=(10, 10), axes_pad=0.1,)
    plot = [x_test[index[0]][:, :, j] for j in range(10)]
    for i in index:
        if k != 0:
            original = [x_test[i][:, :, j] for j in range(10)]
            plot = np.vstack((plot, original))
        reconst = [rec_imgs[i][:, :, j] for j in range(10)]
        plot = np.vstack((plot, reconst))
        k = k + 1
    for ax, im in zip(grid, plot):
        plt.gray()
        ax.imshow(im)
    plt.show()

def create_dataset(width=28, height=28, frames=10, amount=10):
    x_data=[]
    for i in range(amount):
        ani = create_animation_images(width = width, height = height, frames = frames)
        x_data.append(ani)
    return x_data


if __name__ == "__main__":
    balls = Balls(width=28, height=28)
    img = create_picture(balls, 28, 28)

    #data = create_dataset()
    show_dataset([[img]], [[img]])
