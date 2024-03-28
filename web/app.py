import os
import csv
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set the secret key
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if a filename has an allowed extension


@app.route('/')
def index():
    return render_template('index.html')


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


@app.route('/work.html')
def work():
    return render_template('work.html')


@app.route('/crew.html')
def crew():
    return render_template('crew.html')


@app.route('/sugg.html')
def sugg():
    image_names = []
    with open('fav.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_names.extend(row)

    # Get the path to the images directory
    images_dir = os.path.join(app.root_path, 'images')

    return render_template('sugg.html', image_names=image_names, images_dir=images_dir)


@app.route('/cart.html')
def cart():
    image_names = []
    with open('fav.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_names.extend(row)

    # Get the path to the images directory
    images_dir = os.path.join(app.root_path, 'images')

    return render_template('cart.html', image_names=image_names, images_dir=images_dir)


# Route to handle image upload and processing
# Route to handle image upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        # If no file is uploaded, print a message to the console
        print("No file uploaded")
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        # If the filename is empty, return without doing anything
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Process the uploaded file here (you can call your backend code)
        return redirect('/sugg.html')  # Redirect to sugg.html after upload

# Define a route for /upload that only accepts POST requests


@app.route('/upload', methods=['POST'])
def handle_upload():
    return "", 200  # Return an empty response with status code 200



if __name__ == '__main__':
    app.run(debug=True)
