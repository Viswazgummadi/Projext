import os


def find_border(image):
    border_points = []

    # Get image dimensions
    height, width = len(image), len(image[0])

    # Find starting pixel on the border
    for i in range(height):
        if image[i][0] == 1:
            start_point = (i, 0)
            break
    else:
        return border_points  # No border found

    # Define neighbor directions
    directions = [
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1)    # Right
    ]

    # Start contour tracing
    current_point = start_point
    next_point = (current_point[0], current_point[1] + 1)  # Start moving right

    while next_point != start_point:
        border_points.append(current_point)
        found_neighbor = False
        for d in directions:
            neighbor = (current_point[0] + d[0], current_point[1] + d[1])
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and image[neighbor[0]][neighbor[1]] == 1:
                next_point = neighbor
                found_neighbor = True
                break
        if not found_neighbor:
            return border_points  # No neighboring border found
        current_point = next_point

    return border_points


def detect_shirts_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Convert to binary image (thresholding)
            _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # Find border points
            border_points = find_border(binary_img)

            # Draw the border on the original image
            for point in border_points:
                img[point[0]][point[1]] = 255  # Set border pixel to white

            # Save the bordered image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)


# Specify input and output folders
input_folder = 'grayscaled'  # Assuming grayscaled folder contains grayscale images
output_folder = 'bordered'

# Detect shirts in images in the input folder and save them in the output folder
detect_shirts_in_folder(input_folder, output_folder)
