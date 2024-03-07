import cv2

# Load the image
image = cv2.imread('1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a basic thresholding to create a binary image
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours and check for shapes resembling shirts
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(approx) >= 4:
        print("Shirt detected!")
        break
else:
    print("No shirt detected.")
