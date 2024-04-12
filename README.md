# Project Report

## Project Title: Image Processing and Computer Vision Pipeline

### Context:
Our project is aimed at developing an image processing and computer vision pipeline that can analyze images, detect borders, perform color clustering, and classify images based on color features. The pipeline consists of several interconnected modules, each focusing on a specific aspect of image processing and analysis.

### Part 2: Border Detection
In Part 2 of the project, our focus was on implementing border detection techniques. We utilized Canny edge detection algorithm to identify edges in images. Subsequently, we developed a method to extract a rectangular region around detected edges, providing a focused area for further analysis.

### Part 3: Color Clustering and Classification
Building upon the border detection results obtained in Part 2, Part 3 of the project involved color clustering and image classification. We implemented KMeans clustering algorithm to group similar colors in the extracted regions. The clustered colors were then saved into a CSV file (`colors.csv`) for further analysis. Furthermore, a classifier was developed using the k-nearest neighbors (KNN) algorithm to categorize images based on the clustered colors. The classifier was trained on a dataset consisting of images and their corresponding color clusters. Additionally, to enhance the diversity of color suggestions, we incorporated a mechanism to generate five random color suggestions alongside the top five suggestions from the classifier.

### Part 4: Web Application Implementation
The final part of the project focused on implementing a web application to showcase the functionalities developed in the previous stages. Using Flask as the backend framework, we integrated the color clustering and classification algorithms into the web application. Users can upload images, which are then processed by the pipeline to generate color suggestions based on clustering and classification. The web interface provides an intuitive platform for users to interact with the image processing functionalities.

### Conclusion:
In conclusion, our image processing and computer vision pipeline successfully integrates various techniques, including border detection, color clustering, and image classification, to analyze and classify images based on color features. The development of the web application provides a user-friendly interface for individuals to utilize these functionalities. Moving forward, further enhancements and optimizations can be made to improve the accuracy and performance of the pipeline.

For detailed implementation and code, please refer to the respective notebooks and files in the project repository.
