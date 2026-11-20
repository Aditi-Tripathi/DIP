#11
import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread(r"C:\Users\Devesh Tripathi\OneDrive\Desktop\suryakiran.jpeg", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), dtype=np.uint8)
erosion = cv2.erode(binary, kernel, iterations=1)
dilation = cv2.dilate(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

fig, ax = plt.subplots(2, 3, figsize=(11, 7))
for a, im, title in zip(ax.flat,
    [binary, erosion, dilation, opening, closing],
    ["Binary Input", "Erosion", "Dilation", "Opening", "Closing"]):
    a.imshow(im, cmap="gray"); a.set_title(title); a.axis("off")
plt.tight_layout(); plt.show()
