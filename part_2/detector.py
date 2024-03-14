import cv2
import numpy as np
import os
import tensorflow as tf

# Function to preprocess images


def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)

    # Threshold intensities
    edges[edges > 100] = 255
    edges[edges <= 100] = 0

    return edges


# Load images from 'shirts' folder
shirts_folder = 'shirts'
shirts_images = []
for filename in os.listdir(shirts_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(shirts_folder, filename)
        preprocessed_image = preprocess_image(image_path)
        shirts_images.append(preprocessed_image)

# Prepare data for training
X_train = np.array(shirts_images)
y_train = np.ones((len(shirts_images),), dtype=int)  # All images are shirts

# Define and train a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

# Function to detect shirts in images of people wearing shorts


def detect_shirts(image_path):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)

    # Threshold intensities
    edges[edges > 100] = 255
    edges[edges <= 100] = 0

    # Make predictions using the trained model
    prediction = model.predict(np.expand_dims(edges, axis=0))[0]

    # Return the boundary image (edges)
    return edges


# Test the shirt detection function on images from 'ppl_shirts' folder
ppl_shirts_folder = 'ppl_shirts'
for filename in os.listdir(ppl_shirts_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(ppl_shirts_folder, filename)
        boundary_image = detect_shirts(image_path)
        cv2.imwrite(f"boundary_{filename}", boundary_image)
