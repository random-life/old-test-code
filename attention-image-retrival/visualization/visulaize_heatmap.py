import cv2
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('./data/all_souls_000151.jpg')

data = cv2.imread('./sw/all_souls_000151.jpg', 1)
gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots(1)
implot = ax.imshow(img)
heatmap = ax.pcolor(gray_image, alpha=0.1)
plt.axis('off')
plt.savefig("./sw/test3.png")