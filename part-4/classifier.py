import csv
import numpy as np

# Load data from output.csv
files = "output.csv"
outs = []
c1 = []
c2 = []
c3 = []

with open(files, 'r') as file:
    csv_reader = csv.reader(file)
    count = 0
    for row in csv_reader:
        if count > 0:
            outs.append(row[0])
            cc1 = row[1].strip('[]').split()
            c1.append([int(element) for element in cc1])
            cc2 = row[2].strip('[]').split()
            c2.append([int(element) for element in cc2])
            cc3 = row[3].strip('[]').split()
            c3.append([int(element) for element in cc3])
        count += 1
print(cc1)
outs = np.array(outs)
c1 = np.array(c1)
c2 = np.array(c2)
c3 = np.array(c3)

# Combine color channels into feature vectors
X_train = np.hstack((c1, c2, c3))
y_train = np.array(outs)

# Define KNN classifier
class KNNClassifier:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):

        
        def cal_dis(inp1, inp2):
                inp1_reshaped = np.reshape(inp1, inp2.shape)
                return np.linalg.norm(inp1_reshaped - inp2)


        distances = [cal_dis(new_input, x) for x in self.X_train]
        closest_indices = np.argsort(distances)[:5]
        nearest_labels = self.y_train[closest_indices]

        return np.array(nearest_labels)

# Instantiate and train the KNN classifier
knn_classifier = KNNClassifier(k=5)
knn_classifier.fit(X_train, y_train)

# New input data
new_input = np.array([[0, 0, 0], [48, 31, 20], [250, 250, 250]])

# Reshape new_input correctly


# Predict labels for new input using KNN
knn_predictions = knn_classifier.predict(new_input)

# Save predicted labels to image_names.csv
with open("image_names.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(map(lambda x: [x], knn_predictions))

print("Predicted labels for the input data points:", knn_predictions)
