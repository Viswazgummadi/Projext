import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # PIL is used to load the image, not for any processing


def load_image(image_path):
    """
    Load the image using PIL.
    """
    return np.array(Image.open(image_path))


def rgb_to_gray(image):
    """
    Convert the image to grayscale.
    """
    # Apply luminance method
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


def vertical_edge_detection(image, kernel):
    """
    Perform vertical edge detection using convolution.
    """
    # Pad the image to handle edge cases
    # image ,   row wise ,  column wise,  mode = const == by default set them to 0
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    # Create 3D arrays for sliding window
    window_height, window_width = kernel.shape
    padded_height, padded_width = padded_image.shape
    strides = padded_image.strides
    shape = (padded_height - window_height + 1, padded_width -
             window_width + 1, window_height, window_width)
    strides = padded_image.strides * 2
    windows = np.lib.stride_tricks.as_strided(
        padded_image, shape=shape, strides=strides)

    # Convolve the image with the kernel using vectorized operations
    convolved = np.sum(windows[:, :, :, :, np.newaxis] *
                       kernel[np.newaxis, np.newaxis, :, :, np.newaxis], axis=(2, 3))

    return convolved


def threshold(image, threshold_value):
    """
    Threshold the image to obtain binary edge information.
    """
    return (image > threshold_value).astype(np.uint8) * 255


# Define the edge detection kernel
kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])

# Load the image
# Replace 'your_image.jpg' with the path to your image
image = load_image('your_image.jpg')

# Convert the image to grayscale
gray_image = rgb_to_gray(image)

# Perform vertical edge detection
edge_detected_image = vertical_edge_detection(gray_image, kernel)

# Threshold the result
threshold_value = 50  # You may need to adjust this threshold value based on your image
binary_edges = threshold(edge_detected_image, threshold_value)

# Display the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(1, 2, 2)
# Squeeze to remove the third axis
plt.imshow(binary_edges.squeeze(), cmap='binary')
plt.title('Vertical Edges')

plt.savefig('output_plot.png')
