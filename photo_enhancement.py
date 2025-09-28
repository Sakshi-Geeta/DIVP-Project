#Group project by:
#Sakshi Geeta (220130111115)
#Pankti Padariya (230133111024)
#Oza Tanu Saleengram (220133111020)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gamma correction function
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Read image
img = cv2.imread("input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Histogram equalization on Y channel
ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
y, cr, cb = cv2.split(ycrcb)
y_eq = cv2.equalizeHist(y)
ycrcb_eq = cv2.merge((y_eq, cr, cb))
hist_eq_img = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)

# Gamma correction
gamma_corrected = adjust_gamma(hist_eq_img, gamma=1.2)

# Gaussian blur + sharpening
denoised = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(denoised, -1, kernel)

# Display results in a clean, professional layout
titles = ['Original', 'Histogram Equalized', 'Gamma Corrected', 'Enhanced']
images = [img, hist_eq_img, gamma_corrected, sharpened]

plt.figure(figsize=(16,10))
for i in range(4):
    ax = plt.subplot(2, 2, i+1)
    ax.imshow(images[i])
    ax.set_title(titles[i], fontsize=16, fontweight='bold', color='#333333')
    ax.axis('off')
    # Add a subtle border around each subplot
    for spine in ax.spines.values():
        spine.set_edgecolor('#666666')
        spine.set_linewidth(1.2)

plt.tight_layout(pad=3)
plt.show()

# Save output
cv2.imwrite("enhanced_output.jpg", cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR))

