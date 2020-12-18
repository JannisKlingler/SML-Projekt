import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from skimage import draw
import time
# %%
# Probleme: buggy für größere Step_size (relativ stabil für step_size = 0.025). Wenn der
# Größenunterschied zwischen den Bällen zu groß ist oder wenn es zu viele Bälle gibt ist es
# auch für kleinere Step_size buggy
# Plotten sieht für kleine Bilder schlecht aus. Besser wäre da ein Grauwert.

object_number = 3
picture_size = 200
step_size = 0.05
frames = 100
# Lost die Startposition, Radius und Richtungsvektor so aus, dass es keine Überschneidungen gibt:

radius = [np.random.uniform(0.05, 0.15)]
position = [np.random.uniform(0 + radius[0], 1 - radius[0], size=2)]
velocity = [np.random.uniform(-1, 1, size=2)]
for i in range(object_number - 1):
    counter = 0
    restart = True
    while restart:
        start_radius = np.random.uniform(0.05, 0.15)
        start_position = np.random.uniform(0 + start_radius, 1 - start_radius, size=2)
        start_velocity = np.random.uniform(-1, 1, size=2)
        collisions = np.linalg.norm(start_position - np.array(position),
                                    axis=1) < np.array(radius) + start_radius
        counter += 1
        if sum(collisions) == 0:
            restart = False
            break
        if counter == 100:
            restart = False
            break
    if counter == 100:
        print('Mit diesen Konfigurationen wurden keine passenden Startbedingungen gefunden')
        break
    else:
        radius.append(start_radius)
        position.append(start_position)
        velocity.append(start_velocity)

radius = np.array(radius)
position = np.array(position)
velocity = np.array(velocity)


# print(velocity)


def time_step(step_size, position, velocity, radius):
    position += step_size * velocity

    collision_matrix = np.array([np.linalg.norm(position[i] - position[j]) < radius[i] + radius[j]
                                 for i in range(object_number) for j in range(object_number)]).reshape(object_number, object_number)
    for i in range(object_number):
        for j in range(object_number):
            if (j < i) and collision_matrix[i, j] == True:
                # Berechnet neue Velosity der Bälle, falls es eine Kollision gab
                m1, m2 = radius[i] ** 2, radius[j] ** 2
                x1, x2 = position[i], position[j]
                d = np.linalg.norm(x1 - x2) ** 2
                v1, v2 = velocity[i], velocity[j]

                u1 = v1 - 2 * m2 / (m1 + m2) * np.dot(v1 - v2, x1 - x2) / d * (x1 - x2)
                u2 = v2 - 2 * m1 / (m1 + m2) * np.dot(v2 - v1, x2 - x1) / d * (x2 - x1)

                # Berechnet die Strecke die die Bälle seit der Kollision zurückgelegt haben, kann in seltenen Fällen dazu führen,
                # dass Bälle für ein Frame ineinander geraten.
                def F(r):
                    return np.linalg.norm(x1 - x2 + r * (u1 - u2)) - (radius[i] + radius[j])
                t = fsolve(F, 1)[0]
                velocity[i] = u1
                velocity[j] = u2
                position[i] += 2 * t * u1
                position[j] += 2 * t * u2

    # Kollisionen mit den Wänden modellieren:
    a = position[:, 0] - radius < 0
    b = position[:, 0] + radius > 1
    c = position[:, 1] - radius < 0
    d = position[:, 1] + radius > 1
    if True in a:
        velocity[a, 0] = - velocity[a, 0]
        position[a, 0] = radius[a]
    if True in b:
        velocity[b, 0] = - velocity[b, 0]
        position[b, 0] = 1 - radius[b]
    if True in c:
        velocity[c, 1] = - velocity[c, 1]
        position[c, 1] = radius[c]
    if True in d:
        velocity[d, 1] = - velocity[d, 1]
        position[d, 1] = 1 - radius[d]
    return position, velocity

# Wandelt die Positionen in ein Bild um:


for i in range(frames):
    plt.axis('off')
    arr = np.zeros((picture_size, picture_size))
    position, velocity = time_step(step_size, position, velocity, radius)
    for i in range(object_number):
        ro, co = draw.circle(position[i, 0] * picture_size, position[i, 1] *
                             picture_size, radius=radius[i]*picture_size, shape=arr.shape)
        arr[ro, co] = 1
    plt.imshow(arr, cmap='gray')
    plt.pause(0.02)
    plt.clf()
plt.show()
