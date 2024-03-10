import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os

def extract_dominant_colors(image_path, k=3):
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixels = image_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

def process_images(input_folder, output_csv):
    data = {'image_name': [], 'color1': [], 'color2': [], 'color3': []}

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')): 
            image_path = os.path.join(input_folder, filename)

            dominant_colors = extract_dominant_colors(image_path, k=3)

            data['image_name'].append(filename)
            for i in range(3):
                data[f'color{i+1}'].append(dominant_colors[i])  # Convert tuple to string
         
    df = pd.DataFrame(data)

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_folder = "tshirt"
    output_csv = "output.csv"

    process_images(input_folder, output_csv)
