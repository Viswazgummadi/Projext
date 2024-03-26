import cv2
import numpy as np
import joblib
import csv

# Load the trained Random Forest Classifier
model_path = 'gender_detection_model_knn_k27.model'  # Replace with the path to your .model file
clf = joblib.load(model_path)

# Function to detect gender from a given photo file and save the result in a CSV file
def detect_gender_from_photo(photo_path, output_csv):
    # Read the image
    image = cv2.imread(photo_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Resize the image to match the training data dimensions
    resized_image = cv2.resize(gray_image, (96, 96))

    # Flatten the resized image for the classifier
    flat_image = resized_image.reshape((1, -1))

    # Predict gender
    prediction = clf.predict(flat_image)[0]

    # Map prediction to labels
    gender_label = 1 if prediction == 1 else 0

    # Write result to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([gender_label])

photo_path = 'gender_dataset_face/woman/face_221.jpg'  # Replace with the actual path to your photo
output_csv = 'gender.csv'  # Output CSV file name
detect_gender_from_photo(photo_path, output_csv)
