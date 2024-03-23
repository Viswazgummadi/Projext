from Enigma import *

# Creating dataset for next part
input_folder = "train"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(input_folder, file_name)
        image = cv2.imread(image_path)

        # Process the image (convert to grayscale and apply Canny edge detection)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 30, 190)

        bordered_portion = extract_bordered_portion(image, edges, gray_image)

        # Process the bordered portion (convert to grayscale and apply Canny edge detection)
        if bordered_portion is not None and bordered_portion.size > 0:
            # Convert the bordered portion to grayscale
            bordered_gray = cv2.cvtColor(bordered_portion, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection to the bordered portion
            bordered_edges = cv2.Canny(bordered_gray, 30, 190)

            # Save the processed image
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, bordered_edges)
            print(f"Saved processed image of {file_name}.")

print("Image processing completed.")
