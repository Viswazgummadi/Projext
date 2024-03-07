import os
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

            # Get the width and height of the image
            width, height = image.size

            # Convert the image to grayscale
            grayscale_image = Image.new('L', (width, height))

            # Get the pixel data of the original image
            pixels = list(image.getdata())

            # Convert pixel data to list of lists representing the image
            pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

            # Iterate through each pixel and calculate the grayscale value
            for y in range(height):
                for x in range(width):
                    # Get RGB values of the pixel
                    r, g, b = pixels[y][x]

                    # Calculate the grayscale value using weighted sum
                    gray_value = int(0.21 * r + 0.72 * g + 0.07 * b)

                    # Set the grayscale value for the corresponding pixel in the new image
                    grayscale_image.putpixel((x, y), gray_value)

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
