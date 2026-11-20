#7
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

gray = cv2.imread(
    r"C:\Users\Devesh Tripathi\OneDrive\Desktop\suryakiran.jpeg",
    cv2.IMREAD_GRAYSCALE
)

rng = np.random.default_rng(7)

sp = gray.copy()

n = int(0.02 * gray.size)

ys = rng.integers(0, gray.shape[0], n)
xs = rng.integers(0, gray.shape[1], n)

sp[ys, xs] = 255

ys = rng.integers(0, gray.shape[0], n)
xs = rng.integers(0, gray.shape[1], n)

sp[ys, xs] = 0

gaussian = np.clip(
    gray + rng.normal(0, 22, gray.shape),
    0, 255
).astype(np.uint8)

restored = cv2.blur(gaussian, (5, 5))

sp_removed = cv2.medianBlur(sp, 5)

gaussian_removed = cv2.GaussianBlur(gaussian, (5, 5), 0)

median = cv2.medianBlur(sp, 5)

wiener_img = np.clip(
    wiener(gaussian, (5, 5)),
    0, 255
).astype(np.uint8)

# Display output
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

images = [
    gray,
    sp,
    gaussian,
    sp_removed,
    gaussian_removed,
    wiener_img
]

titles = [
    "Original Image",
    "Salt & Pepper Noise",
    "Gaussian Noise",
    "Median Filter",
    "Gaussian Filter",
    "Wiener Filter"
]

for a, im, title in zip(ax.flat, images, titles):
    a.imshow(im, cmap="gray")
    a.set_title(title)
    a.axis("off")

plt.tight_layout()
plt.show()
