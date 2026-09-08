import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
from pydub import AudioSegment

# Change this path according to your Spyder folder
INPUT_FILE = r"C:\Users\balas\OneDrive\Desktop\input_song.mp3"

OUTPUT_DIR = "audio_results"
ALPHA = 1e-3

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_audio(filename):
    audio = AudioSegment.from_file(filename).set_channels(1)

    samples = np.asarray(
        audio.get_array_of_samples(),
        dtype=np.float32
    )

    max_value = float(
        1 << (8 * audio.sample_width - 1)
    )

    return audio.frame_rate, np.clip(
        samples / max_value, -1, 1
    )


def save_wav(filename, fs, x):
    x = np.asarray(x, dtype=np.float32)

    peak = np.max(np.abs(x))

    if peak > 0:
        x = 0.98 * x / peak

    wavfile.write(
        filename,
        fs,
        (np.clip(x, -1, 1) * 32767).astype(np.int16)
    )


def unit_impulse(length=1):
    h = np.zeros(length, dtype=np.float32)
    h[0] = 1
    return h


def moving_average(length=9):
    h = np.ones(length, dtype=np.float32)
    return h / np.sum(h)


def alternating(length=9):
    h = np.array(
        [1 if i % 2 == 0 else -1 for i in range(length)],
        dtype=np.float32
    )

    return h / np.sum(np.abs(h))


def exponential_decay(length=32, decay=0.18):
    n = np.arange(length, dtype=np.float32)
    h = np.exp(-decay * n)

    return (h / np.sum(np.abs(h))).astype(np.float32)


def short_echo(delay=12, gain=0.45):
    h = np.zeros(delay + 1, dtype=np.float32)

    h[0] = 1
    h[delay] = gain

    return h / np.sum(np.abs(h))


def convolve_audio(x, h):
    return fftconvolve(x, h, mode="same").astype(np.float32)


def inverse_filter(y, h, alpha=ALPHA):
    N = len(y)
    nfft = N + len(h) - 1

    Y = np.fft.rfft(y, n=nfft)
    H = np.fft.rfft(h, n=nfft)

    Hinv = np.conj(H) / (np.abs(H) ** 2 + alpha)

    xhat = np.fft.irfft(Y * Hinv, n=nfft)

    start = (len(h) - 1) // 2

    return xhat[start:start + N].astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def correlation(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)

    return float(
        np.sum(a * b) /
        (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12)
    )


def centroid(x, fs):
    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / fs)

    return float(
        np.sum(f * X) / (np.sum(X) + 1e-12)
    )


def save_plot(original, convolved, recovered, h, fs, name):

    n = min(len(original), int(5 * fs))
    t = np.arange(n) / fs

    fig, ax = plt.subplots(3, 1, figsize=(11, 8))

    ax[0].plot(t, original[:n])
    ax[0].set_title("Original Audio")

    ax[1].plot(t, convolved[:n])
    ax[1].set_title("After Convolution")

    ax[2].plot(t, recovered[:n])
    ax[2].set_title("After Regularized Inverse Filtering")

    for a in ax:
        a.set_xlabel("Time (s)")
        a.set_ylabel("Amplitude")
        a.grid(True, alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, name + "_comparison.png"),
        dpi=180
    )

    plt.close()

    plt.figure(figsize=(9, 4))

    plt.stem(np.arange(len(h)), h, basefmt="k-")

    plt.title("Impulse Response: " + name)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.grid(True, alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, name + "_IR.png"),
        dpi=180
    )

    plt.close()


def main():

    if not os.path.exists(INPUT_FILE):
        print("ERROR: Audio file not found.")
        print("Check the INPUT_FILE path.")
        return

    fs, original = load_audio(INPUT_FILE)

    impulse_responses = {

        "IR1_identity":
            unit_impulse(1),

        "IR2_moving_average":
            moving_average(9),

        "IR3_alternating":
            alternating(9),

        "IR4_exponential_decay":
            exponential_decay(32),

        "IR5_short_echo":
            short_echo(12, 0.45)
    }

    rows = []

    for name, h in impulse_responses.items():

        y = convolve_audio(original, h)

        recovered = inverse_filter(y, h)

        save_wav(
            os.path.join(
                OUTPUT_DIR,
                name + "_convolved.wav"
            ),
            fs,
            y
        )

        save_wav(
            os.path.join(
                OUTPUT_DIR,
                name + "_inverse_filtered.wav"
            ),
            fs,
            recovered
        )

        save_plot(
            original,
            y,
            recovered,
            h,
            fs,
            name
        )

        rows.append((
            name,
            len(h),
            rms(y),
            rms(recovered),
            centroid(original, fs),
            centroid(recovered, fs),
            correlation(original, recovered)
        ))

        print(
            name,
            "IR length =", len(h),
            "correlation =",
            round(rows[-1][-1], 4)
        )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "experiment_results.csv"
        ),
        "w"
    ) as f:

        f.write(
            "IR,IR_Length,Convolved_RMS,"
            "Recovered_RMS,Original_Centroid_Hz,"
            "Recovered_Centroid_Hz,Correlation\n"
        )

        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print("Experiment complete.")
    print("Results are in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()