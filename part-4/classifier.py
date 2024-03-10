import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

outs = np.array(outs)
c1 = np.array(c1)
c2 = np.array(c2)
c3 = np.array(c3)

X = np.hstack((c1, c2, c3))
y = np.array(outs)

new_input = np.array([[0, 0, 0], [48, 31, 205], [250, 250, 250]])

def cal_dis(inp1, inp2):
    d_x = np.sqrt((inp1[0][0] - inp2[0])**2 + (inp1[0][1] - inp2[1])**2 + (inp1[0][2] - inp2[2])**2)
    d_y = np.sqrt((inp1[1][0] - inp2[3])**2 + (inp1[1][1] - inp2[4])**2 + (inp1[1][2] - inp2[5])**2)
    d_z = np.sqrt((inp1[2][0] - inp2[6])**2 + (inp1[2][1] - inp2[7])**2 + (inp1[2][2] - inp2[8])**2)
    dis = np.sqrt(d_x**2 + d_y**2 + d_z**2)
    return dis

distances = [cal_dis(new_input, x) for x in X]
closest_indices = np.argsort(distances)[:5]
closest_labels = y[closest_indices]


with open("image_names.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(map(lambda x: [x], closest_labels))
    
print("Predicted labels for the three closest data points:", closest_labels)
