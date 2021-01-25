# My picture project

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from mpl_toolkits.axes_grid1 import ImageGrid



def create_picture(width=28, height=28, rect=[[7, 5],[20, 20]]):
    ### rect = [left_upper_corner, right_lower_corner]
    ### where left_upper_corner = [x_position, y_position],
    ###       rigt_lower_corner = [x_position, y_position],

    #cast rectangle to sutable format
    rect = np.array(rect)

    #adapt rect to picture
    rect[0][0] = max(rect[0][0], 0)
    rect[1][0] = min(rect[1][0], width - 1)
    rect[0][1] = max(rect[0][1], 0)
    rect[1][1] = min(rect[1][1], height - 1)

    #background
    pixel = 1.0
    pic = np.tile(pixel, (height, width))


    # inner rectangle is black
    i_rect = [np.ceil(rect[0]).astype(int), rect[1].astype('int')]
    pic[i_rect[0][1]:i_rect[1][1], i_rect[0][0]:i_rect[1][0]] = [[0]]

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
    data = create_dataset()
    show_dataset(data, data)
