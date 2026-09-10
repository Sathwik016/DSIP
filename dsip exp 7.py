import cv2
import matplotlib.pyplot as plt
import numpy as np

# Image path
image_path = r"C:\Users\balas\OneDrive\Pictures\Krishna.jpeg"

# Read image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("IMAGE NOT FOUND")
else:
    print("IMAGE FOUND")

    # 1. Image Negation
    negative = 255 - img

    # 2. Thresholding
    _, threshold = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # 3. Gamma Correction
    gamma = 0.5

    gamma_img = np.array(
        255 * (img / 255) ** gamma,
        dtype=np.uint8
    )

    # Display images
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(negative, cmap="gray")
    plt.title("Negative Image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(threshold, cmap="gray")
    plt.title("Threshold Image")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(gamma_img, cmap="gray")
    plt.title("Gamma Corrected Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()