import os
import cv2
import numpy as np
from PIL import Image
import csv

# Function to detect shirt and identify colors


def detect_shirt_and_colors(image_path, save_dir):
    # Load the image
    image = cv2.imread(image_path)

    # Check if image is loaded successfully
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a basic thresholding to create a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours are found
    if contours:
        print(f"Contours found: {len(contours)}")
        # Iterate through the contours and check for shapes resembling shirts
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            print(f"Approximated contour sides: {len(approx)}")
            if len(approx) >= 4:
                print("Shirt detected")

                # Get bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Crop the shirt region from the original image
                shirt_region = image[y:y+h, x:x+w]

                # Save the cropped shirt image
                shirt_filename = os.path.join(
                    save_dir, f"{os.path.basename(image_path)}_shirt.jpg")
                cv2.imwrite(shirt_filename, shirt_region)
                print(f"Shirt image saved: {shirt_filename}")

                # Shirt detected, now identify colors
                colors = identify_colors(image)
                return colors

    # No shirt detected
    print("No shirt detected")
    return None

# Function to identify colors in the shirt


def identify_colors(image):
    # Check if image is None
    if image is None:
        return []

    print("Identifying colors...")  # Debugging print statement

    # Resize the image for faster processing
    resized_image = cv2.resize(image, (150, 150))

    # Convert image from OpenCV BGR format to RGB format
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Reshape the image array to a 2D array of pixels
    pixels = resized_image_rgb.reshape((-1, 3))

    # Calculate histogram of colors
    histogram = {}
    for pixel in pixels:
        color = tuple(pixel)
        if color in histogram:
            histogram[color] += 1
        else:
            histogram[color] = 1

    # Sort colors by frequency
    sorted_colors = sorted(histogram.items(), key=lambda x: -x[1])

    # Extract top colors
    top_colors = [color for color, count in sorted_colors[:3]]

    # Convert colors to hexadecimal format
    hex_colors = ['#%02x%02x%02x' %
                  (color[0], color[1], color[2]) for color in top_colors]

    print("Colors identified:", hex_colors)  # Debugging print statement

    return hex_colors


# Path to save detected shirt images
detected_shirts_dir = '/home/vinny/projext/detected_shirts'

# Create the directory if it doesn't exist
if not os.path.exists(detected_shirts_dir):
    os.makedirs(detected_shirts_dir)

# Path to your images directory
images_dir = '/home/vinny/projext/Tshirt'

# List to store rows of the dataset
dataset = []

# Iterate through images in the directory and detect shirt and colors
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(images_dir, filename)
        print(f"Processing image: {image_path}")
        colors = detect_shirt_and_colors(image_path, detected_shirts_dir)
        if colors:
            dataset.append([filename] + colors)
        else:
            dataset.append([filename, "No shirt detected"])

# Save the dataset to a CSV file
csv_file_path = 'shirt_color_dataset.csv'
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Color 1', 'Color 2',
                    'Color 3'])  # Write header row
    writer.writerows(dataset)

print(f"Dataset created successfully and saved at: {csv_file_path}")
