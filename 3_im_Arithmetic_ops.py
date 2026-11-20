import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.resize(cv2.imread("suryakiran.jpeg"), (500, 500))
img2 = cv2.resize(cv2.imread("iaf.jpeg"), (500, 500))

added = cv2.add(img1, img2)
subtracted = cv2.subtract(img1, img2)

gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mean_value = np.mean(gray)

bright = cv2.add(gray, 50)
dark = cv2.subtract(gray, 50)

print(f"Mean value of image: {mean_value:.2f}")

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

for a, im, title in zip(
    ax.flat,
    [added, subtracted, bright, dark],
    ["Addition", "Subtraction", "Increased Brightness", "Decreased Brightness"]
):
    a.imshow(im, cmap="gray" if im.ndim == 2 else None)
    a.set_title(title)
    a.axis("off")

plt.tight_layout()
plt.show()
