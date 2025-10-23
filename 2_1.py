import cv2
import numpy as np
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt

#υνάρτηση για προσθήκη AWGN με SNR και αρχικοποίηση περιβάλλοντος
def add_awgn(img, snr_db):
    img = img.astype(np.float32) / 255.0
    signal_power = np.mean(img**2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)

#Συνάρτηση για MSE 
def compute_mse(original, processed):
    return mean_squared_error(original, processed)

# Φόρτωση εικόνας 
img = cv2.imread("flowers.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Παράμετροι
snr_values = [10, 15, 18]
kernel_sizes = [5, 7, 9]

results = {}

# Δημιουργία grid
fig, axes = plt.subplots(len(snr_values), len(kernel_sizes)+2, figsize=(15, 10))
fig.suptitle("Μάσκα μέσου όρου σε AWGN", fontsize=16)

for i, snr in enumerate(snr_values):
    noisy_img = add_awgn(img, snr)
    axes[i, 0].imshow(img)
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(noisy_img)
    axes[i, 1].set_title(f"Noisy SNR={snr}dB")
    axes[i, 1].axis("off")

    for j, k in enumerate(kernel_sizes):
        filtered = cv2.blur(noisy_img, (k, k))
        mse = compute_mse(img, filtered)
        results[(snr, k)] = mse

        axes[i, j+2].imshow(filtered)
        axes[i, j+2].set_title(f"Mean {k}x{k}\nMSE={mse:.3f}")
        axes[i, j+2].axis("off")

plt.tight_layout()
plt.show()

# Εκτύπωση αποτελεσμάτων
for key, value in results.items():
    snr, k = key
    print(f"SNR={snr} dB, Kernel={k}x{k} -> MSE={value:.4f}")
