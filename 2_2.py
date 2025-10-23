import cv2
import numpy as np
from skimage.util import random_noise
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Αρχικοποίηση περιβάλλοντος 
# Συνάρτηση για προσθήκη salt & pepper θορύβου
# ορισμός θορύβου. 
def add_salt_pepper(img, prob):
    noisy = random_noise(img.astype(np.float32) / 255.0, mode='s&p', amount=prob)
    noisy = (noisy * 255).astype(np.uint8)
    return noisy


# Συνάρτηση για MSE όμοια με πριν 
def compute_mse(original, processed):
    return mean_squared_error(original, processed)

# Φόρτωση εικόνας όμοια με πριν
img = cv2.imread("flowers.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Πιθανότητα κρουστικού θορύβου από την εκφώνηση
probabilities = [0.10, 0.20, 0.25]
kernel_sizes = [5, 7]

# Λίστα αποτελεσμάτων για δυναμική αποθήκευση
results = {}

# Δημιουργία grid
fig, axes = plt.subplots(len(probabilities), len(kernel_sizes)+2, figsize=(12, 9))
fig.suptitle("Median filter σε Salt & Pepper noise", fontsize=16)

for i, prob in enumerate(probabilities):
    noisy_img = add_salt_pepper(img, prob)
    axes[i, 0].imshow(img)
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(noisy_img)
    axes[i, 1].set_title(f"Noisy p={prob*100:.0f}%")
    axes[i, 1].axis("off")

    for j, k in enumerate(kernel_sizes):
        filtered = cv2.medianBlur(noisy_img, k)
        mse = compute_mse(img, filtered)
        results[(prob, k)] = mse

        axes[i, j+2].imshow(filtered)
        axes[i, j+2].set_title(f"Median {k}x{k}\nMSE={mse:.3f}")
        axes[i, j+2].axis("off")

plt.tight_layout()
plt.show()

# Εκτύπωση αποτελεσμάτων
for key, value in results.items():
    prob, k = key
    print(f"Salt&Pepper p={prob*100:.0f}%, Kernel={k}x{k} -> MSE={value:.4f}")

