import os
import cv2
import numpy as np
from PIL import Image

# Define your CNN model
class CNN:
    def __init__(self):
        self.weights = np.random.randn(150, 150, 3)  # Random weights for simplicity
    
    def predict(self, image):
        # Assuming a simple threshold-based detection for demonstration
        prediction = np.sum(image * self.weights) > 0
        return prediction

# Load and preprocess the dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    for image_file in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (150, 150))  # Resize the image to a fixed size
        image = image / 255.0  # Normalize pixel values
        images.append(image)
        # Assuming the images containing shirts are in a folder named 'shirts'
        if 'shirts' in image_file:
            labels.append(1)  # Label 1 for images containing shirts
        else:
            labels.append(0)  # Label 0 for images not containing shirts
    return np.array(images), np.array(labels)

# Define paths
dataset_path = 'path/to/your/dataset'

# Load and preprocess the dataset
images, labels = load_dataset(dataset_path)

# Create and initialize the CNN model
model = CNN()

# Evaluate the model on the dataset
predictions = [model.predict(image) for image in images]

# Calculate accuracy
accuracy = np.mean(predictions == labels)
print("Accuracy:", accuracy)
