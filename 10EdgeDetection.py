#10
import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread(r"C:\Users\Devesh Tripathi\OneDrive\Desktop\suryakiran.jpeg", cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
canny = cv2.Canny(gray, 80, 160)

results = [gray, np.abs(sobel_x), np.abs(sobel_y),
           np.abs(laplacian), canny]
titles = ["Original Image", "Sobel X", "Sobel Y",
          "Laplacian", "Canny Edge Detection"]
fig, ax = plt.subplots(2, 3, figsize=(11, 7))
for a, im, title in zip(ax.flat, results, titles):
    a.imshow(im, cmap="gray"); a.set_title(title); a.axis("off")
plt.tight_layout(); plt.show()
