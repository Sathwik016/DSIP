import cv2
import numpy as np

# Load the image
image = cv2.imread(r"C:\Users\balas\OneDrive\Pictures\Hare Krishna.jpeg")

if image is None:
    print("Error: Image not found!")
else:
    # Changed filter size
    kernel_size = (7, 7)

    # Create averaging filter
    kernel = np.ones(kernel_size, dtype=np.float32) / 49

    # Apply averaging filter
    smoothed_image = cv2.filter2D(image, -1, kernel)

    # Display results
    
    cv2.imshow("Original Image", image)
    cv2.imshow("7x7 Averaging Filter", smoothed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    import cv2

# Load the image
image = cv2.imread(r"C:\Users\balas\OneDrive\Pictures\Hare Krishna.jpeg")

if image is None:
    print("Error: Image not found!")
else:
    # Changed median filter size
    kernel_size = 7

    # Apply median filter
    smoothed_image = cv2.medianBlur(image, kernel_size)

    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("7x7 Median Filter", smoothed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    import cv2
import numpy as np

# Load the image
image = cv2.imread(r"C:\Users\balas\OneDrive\Pictures\Hare Krishna.jpeg")

if image is None:
    print("Error: Image not found!")
else:
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (7, 7), 1.5)

    # Changed sharpening kernel
    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1,  7, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    # Apply sharpening
    sharpened_image = cv2.filter2D(
        blurred_image, -1, sharpening_kernel
    )

    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.imshow("Sharpened Image", sharpened_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    import cv2
import numpy as np

# Load the image
image = cv2.imread(r"C:\Users\balas\OneDrive\Pictures\Hare Krishna.jpeg")

if image is None:
    print("Error: Image not found!")
else:
    # Changed Gaussian values
    kernel_size = (7, 7)
    sigma = 2.0

    # Create Gaussian kernel
    gaussian_1D = cv2.getGaussianKernel(7, sigma)
    gaussian_kernel = np.outer(gaussian_1D, gaussian_1D)

    # Apply Gaussian smoothing
    smoothed_image = cv2.filter2D(
        image, -1, gaussian_kernel
    )

    # Changed sharpening kernel
    sharpening_kernel = np.array([
        [-1, -1, -1],
        [-1,  7, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    # Apply sharpening
    sharpened_image = cv2.filter2D(
        image, -1, sharpening_kernel
    )

    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Gaussian Smoothed Image", smoothed_image)
    cv2.imshow("Sharpened Image", sharpened_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    