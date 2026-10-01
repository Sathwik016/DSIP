#Discrete Signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, sosfiltfilt

# 1. Sampling parameters
Fs = 16000          # Sampling frequency = 16 kHz
duration = 0.005    # Duration = 5 ms

n = np.arange(0, int(Fs * duration))
t = n / Fs

# 2. Generate four DISCRETE sine waves
f1 = 1000           # 1 kHz
f2 = 2000           # 2 kHz
f3 = 3000           # 3 kHz
f4 = 4000           # 4 kHz

x1 = np.sin(2 * np.pi * f1 * n / Fs)
x2 = np.sin(2 * np.pi * f2 * n / Fs)
x3 = np.sin(2 * np.pi * f3 * n / Fs)
x4 = np.sin(2 * np.pi * f4 * n / Fs)

# 3. Plot four discrete sine waves
plt.figure(figsize=(12, 9))

plt.subplot(4, 1, 1)
plt.stem(n[:80], x1[:80], basefmt=" ")
plt.title("Discrete Sine Wave - 1 kHz")
plt.xlabel("Sample n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.stem(n[:80], x2[:80], basefmt=" ")
plt.title("Discrete Sine Wave - 2 kHz")
plt.xlabel("Sample n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.stem(n[:80], x3[:80], basefmt=" ")
plt.title("Discrete Sine Wave - 3 kHz")
plt.xlabel("Sample n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.stem(n[:80], x4[:80], basefmt=" ")
plt.title("Discrete Sine Wave - 4 kHz")
plt.xlabel("Sample n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. Add all four discrete signals
x = x1 + x2 + x3 + x4

# 5. Plot combined signal in DISCRETE time domain
plt.figure(figsize=(10, 5))

plt.stem(n[:100], x[:100], basefmt=" ")

plt.title("Combined Discrete Signal - Time Domain")
plt.xlabel("Sample n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()

# 6. Frequency Domain using FFT
N = len(x)

X = np.fft.fft(x)

freq = np.fft.fftfreq(N, 1/Fs)

magnitude = np.abs(X) / N

positive = freq >= 0

plt.figure(figsize=(10, 5))

plt.stem(
    freq[positive],
    magnitude[positive],
    basefmt=" "
)

plt.title("Combined Signal - Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(0, 5000)

plt.grid(True)
plt.show()

# 7. Butterworth Digital Low-Pass Filter
order = 4
cutoff_butter = 2000

sos_butter = butter(
    order,
    cutoff_butter,
    btype='lowpass',
    fs=Fs,
    output='sos'
)

butter_output = sosfiltfilt(
    sos_butter,
    x
)

# 8. Butterworth Output - DISCRETE Time Domain
plt.figure(figsize=(10, 5))

plt.stem(
    n[:100],
    butter_output[:100],
    basefmt=" "
)

plt.title("Butterworth Filter Output - Discrete Time Domain")
plt.xlabel("Sample n")
plt.ylabel("Amplitude")

plt.grid(True)
plt.show()

# 9. Butterworth Output - Frequency Domain
B = np.fft.fft(butter_output)

B_magnitude = np.abs(B) / N

plt.figure(figsize=(10, 5))

plt.stem(
    freq[positive],
    B_magnitude[positive],
    basefmt=" "
)

plt.title("Butterworth Filter Output - Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(0, 5000)

plt.grid(True)
plt.show()

# 10. Chebyshev Type-I Digital Low-Pass Filter
cutoff_cheby = 3000
ripple = 0.5

sos_cheby = cheby1(
    order,
    ripple,
    cutoff_cheby,
    btype='lowpass',
    fs=Fs,
    output='sos'
)

cheby_output = sosfiltfilt(
    sos_cheby,
    x
)

# 11. Chebyshev Output - DISCRETE Time Domain
plt.figure(figsize=(10, 5))

plt.stem(
    n[:100],
    cheby_output[:100],
    basefmt=" "
)

plt.title("Chebyshev Filter Output - Discrete Time Domain")
plt.xlabel("Sample n")
plt.ylabel("Amplitude")

plt.grid(True)
plt.show()

# 12. Chebyshev Output - Frequency Domain
C = np.fft.fft(cheby_output)

C_magnitude = np.abs(C) / N

plt.figure(figsize=(10, 5))

plt.stem(
    freq[positive],
    C_magnitude[positive],
    basefmt=" "
)

plt.title("Chebyshev Filter Output - Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(0, 5000)

plt.grid(True)
plt.show()
