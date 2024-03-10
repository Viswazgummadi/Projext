import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def detect_shirt_colors(image_path, num_colors=5):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    
    # Reshape the image to a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Convert to float32 for KMeans
    pixels = np.float32(pixels)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the cluster centers and counts
    cluster_centers = kmeans.cluster_centers_
    cluster_counts = Counter(kmeans.labels_)
    
    # Sort colors by frequency
    colors_sorted = sorted([(count, color) for count, color in zip(cluster_counts.values(), cluster_centers)], reverse=True)
    
    return colors_sorted

# Example usage
image_path = '1004.jpg'
all_colors = detect_shirt_colors(image_path)
for count, color in all_colors:
    print("Color:", color.astype(int), "Frequency:", count)
