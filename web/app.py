import os
import csv
from flask import Flask, render_template

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

if __name__ == '__main__':
    app.run(debug=True)
