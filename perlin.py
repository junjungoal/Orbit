import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from perlin_numpy import generate_perlin_noise_3d
import cv2

np.random.seed(0)
for i in range(200):
    noise = generate_perlin_noise_3d(
        (128, 128, 3), (4, 4, 1), tileable=(False, False, True), 
    )
    noise = (noise+1) /2 * 255
    cv2.imwrite('./textures/perlin_texture_{}.png'.format(i), noise.astype(np.uint8))

