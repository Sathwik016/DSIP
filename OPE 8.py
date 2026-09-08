import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.exposure import match_histograms


# CHANGE THIS PATH
folder = r"C:\Users\balas\OneDrive\Desktop\DSIP"


# Change these names according to your images
image_filenames = [

    "pollen_washed.jpg",

    "radial_spokes.png",

    "pollen_dark.jpg",

    "dark_car.png",

    "ct_scan.png"

]


image_paths = [
    os.path.join(folder, f)
    for f in image_filenames
]


# Histogram Equalization
def compute_he(img):

    return cv2.equalizeHist(img)


# Gaussian Reference Image
def create_gaussian_reference(
        shape,
        mean=128,
        std=45,
        low_val=0,
        high_val=255
):

    np.random.seed(42)

    synth = np.random.normal(
        loc=mean,
        scale=std,
        size=shape
    )

    synth = np.clip(
        synth,
        low_val,
        high_val
    ).astype(np.uint8)

    return synth


# Histogram Matching
def compute_hm(
        source_img,
        reference_img
):

    return match_histograms(
        source_img,
        reference_img
    )


# Processing
results = []


for idx, path in enumerate(image_paths, 1):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )


    if img is None:

        print(
            f"Skipping Image {idx}: "
            f"File not found -> {path}"
        )

        continue


    # Histogram Equalization
    he_img = compute_he(img)


    # Histogram Matching
    if idx == 1:

        ref = create_gaussian_reference(
            img.shape,
            mean=128,
            std=50
        )


    elif idx == 2:

        ref = create_gaussian_reference(
            img.shape,
            mean=120,
            std=40
        )


    elif idx == 3:

        ref = create_gaussian_reference(
            img.shape,
            mean=135,
            std=45
        )


    elif idx == 4:

        ref = create_gaussian_reference(
            img.shape,
            mean=75,
            std=35,
            low_val=5,
            high_val=240
        )


    elif idx == 5:

        ref = create_gaussian_reference(
            img.shape,
            mean=110,
            std=35
        )


    hm_img = compute_hm(
        img,
        ref
    )


    results.append(
        (img, he_img, hm_img, idx)
    )


# Visualization
for orig, he, hm, num in results:

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8)
    )


    # Original Image
    axes[0, 0].imshow(
        orig,
        cmap="gray"
    )

    axes[0, 0].set_title(
        f"Image {num}: Original"
    )

    axes[0, 0].axis("off")


    # Histogram Equalization
    axes[0, 1].imshow(
        he,
        cmap="gray"
    )

    axes[0, 1].set_title(
        "Histogram Equalization"
    )

    axes[0, 1].axis("off")


    # Histogram Matching
    axes[0, 2].imshow(
        hm,
        cmap="gray"
    )

    axes[0, 2].set_title(
        "Histogram Matching"
    )

    axes[0, 2].axis("off")


    # Original Histogram
    axes[1, 0].hist(
        orig.ravel(),
        bins=256,
        range=[0, 256]
    )

    axes[1, 0].set_title(
        "Original Histogram"
    )


    # HE Histogram
    axes[1, 1].hist(
        he.ravel(),
        bins=256,
        range=[0, 256]
    )

    axes[1, 1].set_title(
        "Equalized Histogram"
    )


    # HM Histogram
    axes[1, 2].hist(
        hm.ravel(),
        bins=256,
        range=[0, 256]
    )

    axes[1, 2].set_title(
        "Matched Histogram"
    )


    plt.tight_layout()

    plt.show()