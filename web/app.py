import os
import csv
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def index():
    # Read image names from the CSV file
    image_names = []
    with open('image_names.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_names.extend(row)

    # Get the path to the images directory
    images_dir = os.path.join(app.root_path, 'images')

    return render_template('index.html', image_names=image_names, images_dir=images_dir)


@app.route('/favs.html')
def favs():
    image_names = []
    with open('fav.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_names.extend(row)

    # Get the path to the images directory
    images_dir = os.path.join(app.root_path, 'images')

    return render_template('favs.html', image_names=image_names, images_dir=images_dir)


@app.route('/addToFavorites', methods=['POST'])
def add_to_favorites():
    image = request.json['image']
    # Implement logic to add image to fav.csv
    with open('fav.csv', 'a') as file:
        file.write(image + '\n')
    return 'Image added to favorites'


@app.route('/removeFromFavorites', methods=['POST'])
def remove_from_favorites():
    image = request.json['image']
    # Implement logic to remove image from fav.csv
    with open('fav.csv', 'r') as file:
        lines = file.readlines()
    with open('fav.csv', 'w') as file:
        for line in lines:
            if line.strip() != image:
                file.write(line)
    return 'Image removed from favorites'


if __name__ == '__main__':
    app.run(debug=True)
