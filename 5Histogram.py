import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("suryakiran.jpeg", cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

equalized = cv2.equalizeHist(img)

hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])

# Manual histogram equalization
counts = np.bincount(img.ravel(), minlength=256)

cdf = counts.cumsum()

cdf_min = cdf[cdf > 0].min()

lookup = np.round((cdf - cdf_min) / (img.size - cdf_min) * 255)

manual = lookup[img].astype(np.uint8)

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

ax[0, 0].imshow(img, cmap="gray")
ax[0, 0].set_title("Original")

ax[0, 1].plot(hist)
ax[0, 1].set_title("Original Histogram")

ax[1, 0].imshow(equalized, cmap="gray")
ax[1, 0].set_title("Equalized")

ax[1, 1].plot(hist_eq)
ax[1, 1].set_title("Equalized Histogram")

plt.tight_layout()
plt.show()
