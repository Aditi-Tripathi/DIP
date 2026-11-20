import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Devesh Tripathi\OneDrive\Desktop\suryakiran.jpeg")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

negative = 255 - rgb

horizontal = cv2.flip(rgb, 1)
vertical = cv2.flip(rgb, 0)
both = cv2.flip(rgb, -1)

_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

low, high = gray.min(), gray.max()

contrast = ((gray - low) / (high - low) * 255).astype(np.uint8)

fig, ax = plt.subplots(2, 3, figsize=(12, 8))

for a, im, title in zip(
    ax.flat,
    [negative, horizontal, vertical, both, threshold, contrast],
    ["Negative", "Horizontal Flip", "Vertical Flip", "Both Flip",
     "Thresholding", "Contrast Stretching"]
):
    a.imshow(im, cmap="gray" if im.ndim == 2 else None)
    a.set_title(title)
    a.axis("off")

plt.tight_layout()
plt.show()
