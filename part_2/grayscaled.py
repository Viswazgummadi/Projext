import os
import numpy as np
from PIL import Image
import cv2


def remove_small_discontinuous_edges(edge_map, min_contour_area=100):
    # Convert edge map to binary image
    binary_edge_map = np.uint8(edge_map > 0)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to store the areas of contours
    contour_area_mask = np.zeros_like(edge_map)

    # Filter contours based on area
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            # Fill the contour area in the mask
            cv2.drawContours(contour_area_mask, [contour], -1, 255, -1)

    # Use the mask to keep only larger continuous edges
    filtered_edge_map = cv2.bitwise_and(edge_map, contour_area_mask)

    return filtered_edge_map

# Update the enigma function


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

            # Apply Sobel operator to detect edges
            sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            gradient_magnitude = np.uint8(gradient_magnitude)

            # Remove small discontinuous edges
            filtered_edge_map = remove_small_discontinuous_edges(
                gradient_magnitude)

            # Save the filtered edge map
            edge_map_output_path = os.path.join(
                edge_map_output_folder, file_name)
            Image.fromarray(filtered_edge_map).save(edge_map_output_path)
            print(f"Processed {file_name} and saved as {edge_map_output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")


# Specify the input folder and output folder for edge maps
input_folder = "train"
edge_map_output_folder = "edge_maps_filtered"

# Process images and detect edges
enigma(input_folder, edge_map_output_folder)
