import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("suryakiran.jpeg", cv2.IMREAD_GRAYSCALE)

mask = np.zeros(img.shape, dtype=np.uint8)

cv2.circle(mask, (img.shape[1]//2, img.shape[0]//2), 250, 255, -1)

and_img = cv2.bitwise_and(img, mask)
or_img = cv2.bitwise_or(img, mask)
intersection = cv2.bitwise_and(img, mask)

watermark = np.zeros(img.shape, dtype=np.uint8)

cv2.putText(watermark, "INDIA", (60, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 6)

xor_img = cv2.bitwise_xor(img, watermark)
not_img = cv2.bitwise_not(img)

for title, result in [
    ("AND", and_img),
    ("OR", or_img),
    ("Intersection", intersection),
    ("XOR Watermark", xor_img),
    ("NOT", not_img)
]:
    plt.figure()
    plt.imshow(result, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.show()
