import os
import numpy as np
from PIL import Image
import cv2


def remove_small_discontinuous_edges(edge_map, min_contour_area=100):
    # Convert edge map to binary image
    binary_edge_map = np.uint8(edge_map > 0)

    # Find contours in the binary image
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


def enigma(input_folder, grayscale_output_folder, blurred_output_folder, sobel_x_output_folder, sobel_y_output_folder, edge_map_output_folder, pythogorized_output_folder):
    # Create the output folders if they don't exist
    for folder in [grayscale_output_folder, blurred_output_folder, sobel_x_output_folder, sobel_y_output_folder, edge_map_output_folder, pythogorized_output_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Get the list of files in the input folder
    files = os.listdir(input_folder)

    for file_name in files:
        # Construct the full path of the input file
        input_path = os.path.join(input_folder, file_name)

        try:
            # Open the image
            image = Image.open(input_path)

            # Convert the image to grayscale using NumPy
            grayscale_image = np.array(image.convert('L'))

            # Construct the full path of the output file for grayscale image
            grayscale_output_path = os.path.join(
                grayscale_output_folder, file_name)

            # Save the grayscale image
            Image.fromarray(grayscale_image).save(grayscale_output_path)
            print(
                f"Converted {file_name} to grayscale and saved as {grayscale_output_path}")

            # Apply Gaussian blur to the grayscale image
            blurred_image = apply_gaussian_blur(grayscale_image)

            # Construct the full path of the output file for blurred image
            blurred_output_path = os.path.join(
                blurred_output_folder, file_name)

            # Save the blurred image
            Image.fromarray(blurred_image).save(blurred_output_path)
            print(
                f"Applied Gaussian blur to {file_name} and saved as {blurred_output_path}")

            # Apply Sobel X and Sobel Y operators to the blurred image
            sobel_x_image = apply_sobel_x(blurred_image)
            sobel_y_image = apply_sobel_y(blurred_image)

            # Construct the full paths of the output files for Sobel X and Sobel Y images
            sobel_x_output_path = os.path.join(
                sobel_x_output_folder, file_name)
            sobel_y_output_path = os.path.join(
                sobel_y_output_folder, file_name)

            # Save the Sobel X and Sobel Y images
            Image.fromarray(sobel_x_image).save(sobel_x_output_path)
            Image.fromarray(sobel_y_image).save(sobel_y_output_path)
            print(
                f"Applied Sobel X operator to {file_name} and saved as {sobel_x_output_path}")
            print(
                f"Applied Sobel Y operator to {file_name} and saved as {sobel_y_output_path}")

            # Calculate the gradient magnitude image
            gradient_magnitude = calculate_gradient_magnitude(
                sobel_x_image, sobel_y_image)

            # Save the gradient magnitude image to the "pythogorized" folder
            pythogorized_output_path = os.path.join(
                pythogorized_output_folder, file_name)
            Image.fromarray(gradient_magnitude).save(pythogorized_output_path)
            print(
                f"Calculated gradient magnitude and saved as {pythogorized_output_path}")

            # Apply thresholding to detect edges
            strong_edges, weak_edges = apply_threshold(
                gradient_magnitude, low_threshold=100, high_threshold=200)

            # Apply edge tracking to connect weak edges to strong edges
            edge_map = edge_tracking(strong_edges, weak_edges)

            # Remove small discontinuous edges
            filtered_edge_map = remove_small_discontinuous_edges(edge_map)

            # Construct the full path of the output file for filtered edge map
            filtered_edge_map_output_path = os.path.join(
                edge_map_output_folder, file_name)

            # Save the filtered edge map
            Image.fromarray(filtered_edge_map.astype(np.uint8)
                            ).save(filtered_edge_map_output_path)
            print(
                f"Detected edges and saved as {filtered_edge_map_output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")


def apply_gaussian_blur(image):
    # Define the Gaussian blur kernel
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / \
        24  # Normalize the kernel

    # Apply convolution using NumPy's convolve function
    blurred_image = np.zeros_like(image)
    blurred_image = np.convolve(
        image.flatten(), kernel.flatten(), mode='same').reshape(image.shape)

    return blurred_image.astype(np.uint8)


def apply_sobel_x(image):
    # Sobel X operator
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Apply convolution using NumPy's convolve function
    sobel_x_image = np.zeros_like(image)
    sobel_x_image = np.convolve(
        image.flatten(), sobel_x_kernel.flatten(), mode='same').reshape(image.shape)

    return sobel_x_image.astype(np.uint8)


def apply_sobel_y(image):
    # Sobel Y operator
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply convolution using NumPy's convolve function
    sobel_y_image = np.zeros_like(image)
    sobel_y_image = np.convolve(
        image.flatten(), sobel_y_kernel.flatten(), mode='same').reshape(image.shape)

    return sobel_y_image.astype(np.uint8)


def calculate_gradient_magnitude(sobel_x_image, sobel_y_image):
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x_image.astype(
        np.float32)**2 + sobel_y_image.astype(np.float32)**2)

    return gradient_magnitude.astype(np.uint8)


def apply_threshold(image, low_threshold, high_threshold):
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    return strong_edges, weak_edges


def edge_tracking(strong_edges, weak_edges):
    edge_map = np.zeros_like(strong_edges, dtype=np.uint8)
    edge_map[strong_edges] = 255

    # Define the 8-connectivity kernel for edge tracking
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Apply convolution to check for connectivity with strong edges
    convolved = np.convolve(weak_edges.flatten(),
                            kernel.flatten(), mode='same')

    # Reshape the convolution result to match the shape of the weak_edges array
    convolved_reshaped = convolved.reshape(weak_edges.shape)

    # Update the edge map to include pixels connected to strong edges
    edge_map[(convolved_reshaped > 0) & strong_edges] = 255

    return edge_map


if __name__ == "__main__":
    # Specify the input folder and output folders
    input_folder = "train"
    grayscale_output_folder = "grayscaled"
    blurred_output_folder = "blurred"
    sobel_x_output_folder = "sobel_x"
    sobel_y_output_folder = "sobel_y"
    edge_map_output_folder = "edge_maps"
    pythogorized_output_folder = "pythogorized"

    # Convert images to grayscale and detect edges
    enigma(input_folder, grayscale_output_folder, blurred_output_folder,
           sobel_x_output_folder, sobel_y_output_folder, edge_map_output_folder, pythogorized_output_folder)
