import os
import matplotlib.pyplot as plt
import numpy as np

from pydub import AudioSegment
from scipy.signal import correlate
from scipy.stats import pearsonr


# CHANGE THIS PATH
folder = r"C:\Users\balas\OneDrive\Desktop\DSIP26"


# Find audio files
audio_files = [
    f
    for f in os.listdir(folder)
    if f.lower().endswith(
        (".mp3", ".wav", ".m4a", ".flac")
    )
]


print("Audio files found:")

for i, f in enumerate(audio_files, 1):
    print(i, f)


# Original audio filename
original_file = "the_mountain-reels-song-576585.mp3"


if original_file not in audio_files:
    raise FileNotFoundError(
        "Original song was not found."
    )


other_files = [
    f for f in audio_files
    if f != original_file
]


if len(other_files) < 2:
    raise FileNotFoundError(
        "You need at least 3 audio files."
    )


karaoke_file = other_files[0]
different_file = other_files[1]


print("\nFiles selected:")

print("Original :", original_file)
print("Karaoke :", karaoke_file)
print("Different:", different_file)


# Audio paths
original_path = os.path.join(
    folder,
    original_file
)

karaoke_path = os.path.join(
    folder,
    karaoke_file
)

different_path = os.path.join(
    folder,
    different_file
)


# Load audio
original = AudioSegment.from_file(original_path)

karaoke = AudioSegment.from_file(karaoke_path)

different = AudioSegment.from_file(different_path)


# Sample rate
sample_rate = 8000


# Convert to mono and sample rate
original = original.set_channels(1).set_frame_rate(sample_rate)

karaoke = karaoke.set_channels(1).set_frame_rate(sample_rate)

different = different.set_channels(1).set_frame_rate(sample_rate)


# Convert to arrays
original_samples = np.array(
    original.get_array_of_samples()
).astype(float)

karaoke_samples = np.array(
    karaoke.get_array_of_samples()
).astype(float)

different_samples = np.array(
    different.get_array_of_samples()
).astype(float)


# Z-score normalization
original_samples = (
    original_samples - np.mean(original_samples)
) / np.std(original_samples)


karaoke_samples = (
    karaoke_samples - np.mean(karaoke_samples)
) / np.std(karaoke_samples)


different_samples = (
    different_samples - np.mean(different_samples)
) / np.std(different_samples)


# Common length
minimum_length = min(
    len(original_samples),
    len(karaoke_samples),
    len(different_samples)
)


original_samples = original_samples[:minimum_length]

karaoke_samples = karaoke_samples[:minimum_length]

different_samples = different_samples[:minimum_length]


# Pearson Correlation
r_original_karaoke, _ = pearsonr(
    original_samples,
    karaoke_samples
)

r_original_different, _ = pearsonr(
    original_samples,
    different_samples
)

r_karaoke_different, _ = pearsonr(
    karaoke_samples,
    different_samples
)


# Normalized Cross Correlation
def normalized_cross_correlation(a, b):

    result = correlate(a, b, mode="full")

    result = result / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    return result


cross_original_karaoke = normalized_cross_correlation(
    original_samples,
    karaoke_samples
)

cross_original_different = normalized_cross_correlation(
    original_samples,
    different_samples
)

cross_karaoke_different = normalized_cross_correlation(
    karaoke_samples,
    different_samples
)


# Maximum correlation
max_original_karaoke = np.max(
    np.abs(cross_original_karaoke)
)

max_original_different = np.max(
    np.abs(cross_original_different)
)

max_karaoke_different = np.max(
    np.abs(cross_karaoke_different)
)


# Display results
print("\n========================================")
print("PEARSON CORRELATION")
print("========================================")

print(
    f"Original vs Karaoke : "
    f"{r_original_karaoke:.4f}"
)

print(
    f"Original vs Different : "
    f"{r_original_different:.4f}"
)

print(
    f"Karaoke vs Different : "
    f"{r_karaoke_different:.4f}"
)


print("\n========================================")
print("MAX NORMALIZED CROSS-CORRELATION")
print("========================================")

print(
    f"Original vs Karaoke : "
    f"{max_original_karaoke:.4f}"
)

print(
    f"Original vs Different : "
    f"{max_original_different:.4f}"
)

print(
    f"Karaoke vs Different : "
    f"{max_karaoke_different:.4f}"
)


# Analysis
highest = max(
    [
        ("Original vs Karaoke", max_original_karaoke),
        ("Original vs Different", max_original_different),
        ("Karaoke vs Different", max_karaoke_different)
    ],
    key=lambda x: x[1]
)


print("\n========================================")
print("ANALYSIS")
print("========================================")

print(
    f"Highest cross-correlation: "
    f"{highest[0]} ({highest[1]:.4f})"
)


# Plot Cross Correlation
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)

plt.plot(cross_original_karaoke)

plt.title("Original vs Karaoke")

plt.xlabel("Lag")
plt.ylabel("Correlation")


plt.subplot(3, 1, 2)

plt.plot(cross_original_different)

plt.title("Original vs Different Song")

plt.xlabel("Lag")
plt.ylabel("Correlation")


plt.subplot(3, 1, 3)

plt.plot(cross_karaoke_different)

plt.title("Karaoke vs Different Song")

plt.xlabel("Lag")
plt.ylabel("Correlation")


plt.tight_layout()

plt.show()


# Pearson Bar Chart
labels = [
    "Original-Karaoke",
    "Original-Different",
    "Karaoke-Different"
]

values = [
    r_original_karaoke,
    r_original_different,
    r_karaoke_different
]


plt.figure(figsize=(9, 6))

plt.bar(labels, values)

plt.axhline(
    0,
    linestyle="--",
    color="black",
    alpha=0.7
)

plt.title(
    "Pearson Correlation Between Audio Tracks"
)

plt.xlabel("Audio Track Pair")

plt.ylabel("Correlation Coefficient")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()