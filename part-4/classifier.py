import csv
import numpy as np

def file_open(file):
    outs = []
    ratings = []
    c1 = []
    c2 = []
    c3 = []

    with open(file, 'r') as file:
        csv_reader = csv.reader(file)
        count = 0
        for row in csv_reader:
            if count > 0:
                outs.append(row[0])
                ratings.append(int(row[4]))
                cc1 = row[1].strip('[]').split()
                c1.append([int(element) for element in cc1])
                cc2 = row[2].strip('[]').split()
                c2.append([int(element) for element in cc2])
                cc3 = row[3].strip('[]').split()
                c3.append([int(element) for element in cc3])
            count += 1

    outs = np.array(outs)
    ratings = np.array(ratings)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)

    X = np.hstack((c1, c2, c3))
    y = np.array(outs)
    
    return X,y,ratings

def recommend(X, y, ratings, new_input):
    file="gender.csv"
    with open(file, 'r') as file:
        csv_reader_gender = csv.reader(file) 
        for row in csv_reader_gender:
            if row[1]==0:
                file_open(male_csv)
            else:
                file_open(female_csv)    
        
           
    distances = [cal_dis(new_input, x) for x in X]
    weighted_distances = np.array(distances) * (1 / ratings)
    closest_indices = np.argsort(weighted_distances)[:5]
    closest_labels = y[closest_indices]
    return closest_labels

def cal_dis(inp1, inp2):
    inp1 = np.array(inp1)  
    d_x = np.sqrt((inp1[0][0] - inp2[0])**2 + (inp1[0][1] - inp2[1])**2 + (inp1[0][2] - inp2[2])**2)
    d_y = np.sqrt((inp1[0][3] - inp2[3])**2 + (inp1[0][4] - inp2[4])**2 + (inp1[0][5] - inp2[5])**2)
    d_z = np.sqrt((inp1[0][6] - inp2[6])**2 + (inp1[0][7] - inp2[7])**2 + (inp1[0][8] - inp2[8])**2)
    dis = np.sqrt(d_x**2 + d_y**2 + d_z**2)
    return dis

male_csv="male.csv"
female_csv="female.csv"

file = "input_color.csv"
new1 = []
new2 = []
new3 = []
new_input = []
with open(file, 'r') as file:
    csv_reader_input = csv.reader(file)
    count = 0
    for row in csv_reader_input:
        if count > 0:
            newi_1 = row[1].strip('[]').split()
            new1.append([int(element) for element in newi_1])
            newi_2 = row[2].strip('[]').split()
            new2.append([int(element) for element in newi_2])
            newi_3 = row[3].strip('[]').split()
            new3.append([int(element) for element in newi_3])
        count += 1

c1 = np.array(new1)
c2 = np.array(new2)
c3 = np.array(new3)
new_input= np.hstack((new1,new2,new3))


                

closest_labels = recommend(X, y, ratings, new_input)

with open("image_names.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(map(lambda x: [x], closest_labels))

print("Predicted labels for the three closest data points:", closest_labels)
