from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('your_meme.png')
lol = (np.all(img == [0,0,0],axis=2))
plt.matshow(lol)
plt.show()