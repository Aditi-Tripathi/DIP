import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(r"C:\Users\Devesh Tripathi\OneDrive\Desktop\suryakiran.jpeg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img, (900, 1200))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
b, g, r = cv2.split(img)
reconstructed = cv2.merge((b,g,r))

data = np.zeros((200,200), dtype = np.uint8)
data[::50,:] = 255
data[:,::50] = 255
cv2.imwrite("image_2_D.png", data)

fig, ax = plt.subplots(3,3, figsize = (12, 8))
items = [ rgb, cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), gray, reconstructed, bw, b,g,r, data]

for a, im in zip(ax.flat, items):
    a.imshow(im, cmap= "gray" if im.ndim==2 else None)
    a.axis("off")
plt.tight_layout()
plt.show()
