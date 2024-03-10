import os
import numpy as np
from PIL import Image
import cv2


def enigma(input_folder, edge_map_output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(edge_map_output_folder):
        os.makedirs(edge_map_output_folder)

    # Get the list of files in the input folder
    files = os.listdir(input_folder)

    for file_name in files:
        # Construct the full path of the input file
        input_path = os.path.join(input_folder, file_name)

        try:
            # Open the image
            image = Image.open(input_path)

            # Convert the image to grayscale
            grayscale_image = np.array(image.convert('L'))

            # Apply Gaussian blur
            blurred_image = cv2.GaussianBlur(grayscale_image, (11, 11), 0)

            # Apply Sobel operator to detect edges
            sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            gradient_magnitude = np.uint8(gradient_magnitude)

            # Apply morphological operations to get only the borders
            gradient_magnitude = cv2.Canny(gradient_magnitude, 100, 200)

            # Save the edge map
            edge_map_output_path = os.path.join(
                edge_map_output_folder, file_name)
            Image.fromarray(gradient_magnitude).save(edge_map_output_path)
            print(f"Processed {file_name} and saved as {edge_map_output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")


# Specify the input folder and output folder for edge maps
input_folder = "train"
edge_map_output_folder = "edge_maps_filtered"

# Process images and detect edges
enigma(input_folder, edge_map_output_folder)
