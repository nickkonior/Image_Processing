import cv2
import numpy as np
import matplotlib.pyplot as plt


# Δική μου συνάρτηση 
def my_hist_equalization(img):
    """
    Υλοποίηση εξίσωσης ιστογράμματος από την αρχή.
    Είσοδος: img (εικόνα 8-bit grayscale)
    Έξοδος: equalized (εικόνα μετά από εξίσωση)
    """
    # Υπολογισμός ιστογράμματος
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    # Υπολογισμός σωρευτικής συνάρτησης κατανομής (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # για εμφάνιση
    
    # Κανονικοποίηση CDF -> 0-255
    cdf_m = np.ma.masked_equal(cdf, 0)  # αγνόησε τα μηδενικά
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Χαρτογράφηση παλιών pixel -> νέα
    equalized = cdf_final[img]
    return equalized


# Συνάρτηση: Υπολογισμός DCT2 (έτοιμη)
def dct2(img):
    """
    Υπολογισμός 2D Discrete Cosine Transform (DCT2).
    Χρησιμοποιείται για ανάλυση συχνοτήτων.
    """
    return cv2.dct(np.float32(img)/255.0)

def plot_results(img, eq_my, eq_cv, title):
    """
    Εμφάνιση συγκριτικών αποτελεσμάτων:
    - αρχικό vs δικό μου eq vs OpenCV eq
    - ιστογράμματα
    - φάσμα DCT2
    """
    # DCT2
    dct_original = np.log(np.abs(dct2(img)) + 1)
    dct_my = np.log(np.abs(dct2(eq_my)) + 1)
    dct_cv = np.log(np.abs(dct2(eq_cv)) + 1)
    
    # Ιστόγραμμα
    hist_orig, _ = np.histogram(img.flatten(), 256, [0,256])
    hist_my, _ = np.histogram(eq_my.flatten(), 256, [0,256])
    hist_cv, _ = np.histogram(eq_cv.flatten(), 256, [0,256])
    
    plt.figure(figsize=(14,10))
    plt.suptitle(f"Εξίσωση Ιστογράμματος: {title}", fontsize=16)
    
    # Εικόνες
    plt.subplot(3,3,1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis("off")
    plt.subplot(3,3,2); plt.imshow(eq_my, cmap='gray'); plt.title("My Equalization"); plt.axis("off")
    plt.subplot(3,3,3); plt.imshow(eq_cv, cmap='gray'); plt.title("OpenCV Equalization"); plt.axis("off")
    
    # Ιστογράμματα
    plt.subplot(3,3,4); plt.plot(hist_orig, color='b'); plt.title("Hist Original")
    plt.subplot(3,3,5); plt.plot(hist_my, color='g'); plt.title("Hist My Equalization")
    plt.subplot(3,3,6); plt.plot(hist_cv, color='r'); plt.title("Hist OpenCV Equalization")
    
    # DCT2
    plt.subplot(3,3,7); plt.imshow(dct_original, cmap='gray'); plt.title("DCT2 Original"); plt.axis("off")
    plt.subplot(3,3,8); plt.imshow(dct_my, cmap='gray'); plt.title("DCT2 My Equalization"); plt.axis("off")
    plt.subplot(3,3,9); plt.imshow(dct_cv, cmap='gray'); plt.title("DCT2 OpenCV Equalization"); plt.axis("off")
    
    plt.tight_layout()
    plt.show()


image_files = ["im1.jpg", "im2.jpg"]

for file in image_files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    eq_my = my_hist_equalization(img)
    eq_cv = cv2.equalizeHist(img)
    
    # Σύγκριση αποτελεσμάτων
    plot_results(img, eq_my, eq_cv, file)
