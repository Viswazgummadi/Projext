import os
import numpy as np
from PIL import Image


def convert_to_grayscale(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

            # Create a PIL image from the grayscale NumPy array
            grayscale_image = Image.fromarray(grayscale_image)

            # Construct the full path of the output file
            output_path = os.path.join(output_folder, file_name)

            # Save the grayscale image
            grayscale_image.save(output_path)
            print(
                f"Converted {file_name} to grayscale and saved as {output_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")


if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = "train"
    output_folder = "grayscaled"

    # Convert images to grayscale
    convert_to_grayscale(input_folder, output_folder)
