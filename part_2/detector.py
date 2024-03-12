import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths to the training and testing data folders
train_data_dir = 'output_images'
test_data_dir = 'test'

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Define the number of training and testing samples
num_train_samples = sum([len(files)
                        for r, d, files in os.walk(train_data_dir)])
num_test_samples = sum([len(files) for r, d, files in os.walk(test_data_dir)])

# Create data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(img_width, img_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Calculate steps_per_epoch
steps_per_epoch = num_train_samples // batch_size

# Train the model
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=10)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(
    test_generator, steps=num_test_samples // batch_size)
print(f'Test Accuracy: {test_accuracy}')

# Use the trained model to predict shirt borders in test images
predictions = model.predict(test_generator)
predicted_labels = [1 if pred >= 0.5 else 0 for pred in predictions]

# Print the predicted labels
print(predicted_labels)
