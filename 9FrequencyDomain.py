#9
import cv2
import numpy as np
import matplotlib.pyplot as plt

gray = cv2.imread(r"C:\Users\Devesh Tripathi\OneDrive\Desktop\suryakiran.jpeg", cv2.IMREAD_GRAYSCALE)

rows, cols = gray.shape

dft = np.fft.fftshift(np.fft.fft2(gray))

magnitude = np.log(np.abs(dft) + 1)

y, x = np.ogrid[:rows, :cols]

distance = np.sqrt((y - rows//2)**2 + (x - cols//2)**2)

radius = 35

low_mask = distance <= radius
high_mask = distance > radius

low_dft = dft * low_mask
high_dft = dft * high_mask

low_image = np.abs(np.fft.ifft2(np.fft.ifftshift(low_dft)))
high_image = np.abs(np.fft.ifft2(np.fft.ifftshift(high_dft)))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

for a, im, title in zip(
    ax,
    [magnitude, low_image, high_image],
    ["FFT Magnitude", "Low Pass (IFFT)", "High Pass (IFFT)"]
):
    a.imshow(im, cmap="gray")
    a.set_title(title)
    a.axis("off")

plt.tight_layout()
plt.show()
