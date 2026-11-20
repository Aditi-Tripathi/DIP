import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("suryakiran.jpeg")

rows, cols = img.shape[:2]

M = np.float32([[1, 0, 100], [0, 1, 50]])

translated = cv2.warpAffine(img, M, (cols, rows))

scaled = cv2.resize(img, None, fx=1.5, fy=1.5)

center = (cols//2, rows//2)

R = cv2.getRotationMatrix2D(center, 45, 1.0)

rotated = cv2.warpAffine(img, R, (cols, rows))

shrunk = cv2.resize(img, None, fx=0.5, fy=0.5)

crop = img[rows//4:3*rows//4, cols//4:3*cols//4]

zoomed = cv2.resize(crop, (cols, rows), interpolation=cv2.INTER_CUBIC)

for title, result in [
    ("Translation", translated),
    ("Scaling", scaled),
    ("Rotation", rotated),
    ("Shrinking", shrunk),
    ("Zooming", zoomed)
]:
    plt.figure()
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

plt.show()
