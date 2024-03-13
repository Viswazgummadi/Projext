import os
import cv2
import numpy as np

# Path to the folder containing the images
input_folder = 'output_images/'
output_folder = 'binaried/'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Threshold value for RGB channels
threshold_value = 100

# Iterate through all files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # Read the binary image
        image_path = os.path.join(input_folder, filename)
        binary_image = cv2.imread(image_path)

        # Invert the binary image
        binary_image = cv2.bitwise_not(binary_image)

        # Create masks for pixels greater and lower than the threshold
        high_intensity_mask = np.all(binary_image > threshold_value, axis=2)
        low_intensity_mask = np.all(binary_image <= threshold_value, axis=2)

        # Set high intensity pixels to a high value
        result_image = np.zeros_like(binary_image)
        result_image[high_intensity_mask] = [255, 255, 255]  # High intensity

        # Set low intensity pixels to a low value
        result_image[low_intensity_mask] = [0, 0, 0]  # Low intensity

        # Save the resulting image in the binaried folder
        output_path = os.path.join(output_folder, f'result_{filename}')
        cv2.imwrite(output_path, result_image)
