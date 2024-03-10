import cv2
import numpy as np

# Read the image
image = cv2.imread('102.jpg')


def rgb_to_grayscale(image_array):
    # Calculate luminance using vectorized operations
    luminance = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])

    # Convert luminance array to uint8
    grayscale_array = luminance.astype(np.uint8)

    return grayscale_array


# Convert the image to grayscale using luminance values
gray = rgb_to_grayscale(image)

# Define Gaussian kernel
kernel_size = 5
sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
kernel = cv2.getGaussianKernel(kernel_size, sigma)
gaussian_kernel = np.outer(kernel, kernel.transpose())

# Apply Gaussian blur using kernel multiplication
blurred = cv2.filter2D(gray, -1, gaussian_kernel)

# Use adaptive thresholding to better segment the image
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Perform Canny edge detection
edges = cv2.Canny(thresh, 30, 150)  # You can adjust these threshold values

# Save the edges image
cv2.imwrite('edges_image.png', edges)


