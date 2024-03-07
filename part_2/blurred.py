import os
import numpy as np
from PIL import Image


def convert_to_grayscale(input_folder, grayscale_output_folder, blurred_output_folder, sobel_x_output_folder, sobel_y_output_folder, pythogorized_output_folder):
    # Create the output folders if they don't exist
    for folder in [grayscale_output_folder, blurred_output_folder, sobel_x_output_folder, sobel_y_output_folder, pythogorized_output_folder]:
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
            grayscale_image = np.dot(np.array(image, dtype=np.float32), [
                                     0.21, 0.72, 0.07]).astype(np.uint8)

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

            # Calculate the hypotenuse of Sobel X and Sobel Y components
            pythogorized_image = calculate_hypotenuse(
                sobel_x_image, sobel_y_image)

            # Construct the full path of the output file for pythogorized image
            pythogorized_output_path = os.path.join(
                pythogorized_output_folder, file_name)

            # Save the pythogorized image
            Image.fromarray(pythogorized_image).save(
                pythogorized_output_path)
            print(
                f"Calculated hypotenuse and saved as {pythogorized_output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")


def apply_gaussian_blur(image):
    # Define the Gaussian blur kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16  # Normalize the kernel

    # Apply convolution using NumPy's convolve function
    blurred_image = np.zeros_like(image)
    blurred_image = np.convolve(
        image.flatten(), kernel.flatten(), mode='same').reshape(image.shape)

    return blurred_image.astype(np.uint8)


def apply_sobel_x(image):
    # Sobel X operator
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    # Apply convolution using NumPy's convolve function
    sobel_x_image = np.zeros_like(image)
    sobel_x_image = np.convolve(
        image.flatten(), sobel_x_kernel.flatten(), mode='same').reshape(image.shape)

    return sobel_x_image.astype(np.uint8)


def apply_sobel_y(image):
    # Sobel Y operator
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    # Apply convolution using NumPy's convolve function
    sobel_y_image = np.zeros_like(image)
    sobel_y_image = np.convolve(
        image.flatten(), sobel_y_kernel.flatten(), mode='same').reshape(image.shape)

    return sobel_y_image.astype(np.uint8)


def calculate_hypotenuse(sobel_x_image, sobel_y_image):
    # Calculate hypotenuse
    pythogorized_image = np.sqrt(
        sobel_x_image.astype(np.float32)**2 + sobel_y_image.astype(np.float32)**2)
    # Normalize to 0-255
    pythogorized_image = (pythogorized_image /
                          np.max(pythogorized_image)) * 255

    return pythogorized_image.astype(np.uint8)


if __name__ == "__main__":
    # Specify the input folder and output folders
    input_folder = "train"
    grayscale_output_folder = "grayscaled"
    blurred_output_folder = "blurred"
    sobel_x_output_folder = "sobel_X"
    sobel_y_output_folder = "sobel_Y"
    pythogorized_output_folder = "pythogorized"

    # Convert images to grayscale and apply Gaussian blur
    convert_to_grayscale(input_folder, grayscale_output_folder,
                         blurred_output_folder, sobel_x_output_folder, sobel_y_output_folder, pythogorized_output_folder)
