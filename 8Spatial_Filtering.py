import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread("suryakiran.jpeg", cv2.IMREAD_GRAYSCALE)

signal_1d = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

kernel_1d = np.array([1, 0, -1])

conv_1d = np.convolve(signal_1d, kernel_1d, mode="same")

print("1-D convolution:", conv_1d)

low_pass_mask = np.ones((3, 3), np.float32) / 9

high_pass_mask = np.array(
    [[0, -1, 0],
     [-1, 4, -1],
     [0, -1, 0]],
    dtype=np.float32
)

low_pass = cv2.filter2D(gray, -1, low_pass_mask)

high_pass = cv2.filter2D(gray, -1, high_pass_mask)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(low_pass, cmap="gray")
ax[0].set_title("3x3 Low Pass Filter")

ax[1].imshow(high_pass, cmap="gray")
ax[1].set_title("3x3 High Pass Filter")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()
