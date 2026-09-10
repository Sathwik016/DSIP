import cv2
import matplotlib.pyplot as plt
import numpy as np

# Image path
image_path = r"C:\Users\balas\OneDrive\Pictures\Krishna.jpeg"

# Read image
img = cv2.imread(image_path)

if img is None:
    print("IMAGE NOT FOUND")
else:
    print("IMAGE FOUND")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Gaussian Filter
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    # Averaging Filter
    average = cv2.blur(img, (5, 5))

    # Median Filter
    median = cv2.medianBlur(img, 5)

    # High Pass Sharpening Filter
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    sharpened = cv2.filter2D(img, -1, kernel)

    # Display
    images = [
        img_rgb,
        cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(average, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(median, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    ]

    titles = [
        "Original",
        "Gaussian Filter",
        "Averaging Filter",
        "Median Filter",
        "Sharpened Image"
    ]

    plt.figure(figsize=(15, 8))

    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
    
    
import cv2
import matplotlib.pyplot as plt

image_path = r"C:\Users\balas\OneDrive\Pictures\Krishna.jpeg"

img = cv2.imread(image_path)

if img is None:
    print("IMAGE NOT FOUND")
else:
    print("IMAGE FOUND")

    # Averaging filter
    average = cv2.blur(img, (5, 5))

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    average = cv2.cvtColor(average, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(average)
    plt.title("Averaging Filter")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
    
import cv2
import matplotlib.pyplot as plt

image_path = r"C:\Users\balas\OneDrive\Pictures\Krishna.jpeg"

img = cv2.imread(image_path)

if img is None:
    print("IMAGE NOT FOUND")
else:
    print("IMAGE FOUND")

    # Median filter
    median = cv2.medianBlur(img, 5)

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    median = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(median)
    plt.title("Median Filter")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
    
    
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r"C:\Users\balas\OneDrive\Pictures\Krishna.jpeg"

img = cv2.imread(image_path)

if img is None:
    print("IMAGE NOT FOUND")
else:
    print("IMAGE FOUND")

    # High-pass sharpening kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    # Apply filter
    sharpened = cv2.filter2D(img, -1, kernel)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sharpened = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    # Display
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened)
    plt.title("Sharpened Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()   