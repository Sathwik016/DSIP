import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Load the image in grayscale
image_path = r"C:\Users\balas\OneDrive\Pictures\Hare Krishna.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not find or open the image at: {image_path}")

# 2. Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# 3. Calculate histograms for both images
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# 4. Display images and histograms in a single clean layout
plt.figure(figsize=(12, 10))

# Subplot 1: Original Image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Subplot 2: Equalized Image
plt.subplot(2, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

# Subplot 3: Original Histogram
plt.subplot(2, 2, 3)
plt.title('Original Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(hist_original, color='blue')
plt.xlim([0, 256])
plt.grid(True)

# Subplot 4: Equalized Histogram
plt.subplot(2, 2, 4)
plt.title('Equalized Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(hist_equalized, color='red')
plt.xlim([0, 256])
plt.grid(True)

plt.tight_layout()
plt.show()