import cv2
import numpy as np
import matplotlib.pyplot as plt

# Φόρτωση εικόνας 
img = cv2.imread("car.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0  # κανονικοποίηση

# Bήμα 1: log μετασχηματισμός
log_img = np.log1p(img)  # log(1+I)

# Βήμα 2: Μετασχηματισμός Φourier 
dft = np.fft.fft2(log_img)
dft_shift = np.fft.fftshift(dft)

# Βήμα 3: Δημιουργία Butterworth high-pass filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
D0 = 60  # cutoff frequency
n = 2    # τάξη φίλτρου

u = np.arange(rows)
v = np.arange(cols)
V, U = np.meshgrid(v, u)
D = np.sqrt((U - crow)**2 + (V - ccol)**2)

# Butterworth high-pass
H_hp = 1 / (1 + (D0 / (D + 1e-5))**(2*n))

#Βήμα 4: Ομοιομορφικό φίλτρο
gammaL = 0.9   # ενίσχυση χαμηλών
gammaH = 1.5   # ενίσχυση υψηλών

H = (gammaH - gammaL) * H_hp + gammaL

#Εφαρμογή φίλτρου 
filtered_shift = H * dft_shift
filtered = np.fft.ifftshift(filtered_shift)
inv_img = np.fft.ifft2(filtered)
inv_img = np.real(inv_img)

# Βήμα 5: exp για να γυρίσουμε πίσω
homomorphic_img = np.expm1(inv_img)
homomorphic_img = np.clip(homomorphic_img, 0, 1)

# Εμφάνιση τελικής εικόνας
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title("Original car.jpg"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(homomorphic_img, cmap="gray"); plt.title("Homomorphic Filtered"); plt.axis("off")
plt.show()
