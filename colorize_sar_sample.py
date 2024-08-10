import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the SAR image (assuming it's in a file called 'sar_image.png')
sar_image = cv2.imread('libcong.jpg', cv2.IMREAD_GRAYSCALE)

# Optionally normalize the image to the 0-255 range
sar_image_normalized = cv2.normalize(sar_image, None, 0, 255, cv2.NORM_MINMAX)

# Apply a vibrant colormap (e.g., 'jet', 'plasma', 'inferno', 'viridis')
colored_image = cv2.applyColorMap(sar_image_normalized, cv2.COLORMAP_PINK)

# Display the original and colorized image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original SAR Image')
plt.imshow(sar_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Colorized SAR Image')
plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

# Optionally, save the colorized image
cv2.imwrite('colorized_sar_image.png', colored_image)
