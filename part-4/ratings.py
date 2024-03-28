import csv
import random

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            data.append(row)
    return data

def save_data(file_path, data):
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data)

def generate_random_ratings(data):
    for row in data:
        rating = random.randint(1, 5)  # Generate random rating between 1 and 5
        row.append(rating)
    return data

def main():
    input_file_path = "output.csv"
    output_file_path = "out.csv"
    
    # Load data from existing CSV file
    data = load_data(input_file_path)
    
    # Generate random ratings for each photo
    data_with_random_ratings = generate_random_ratings(data)
    
    # Save data with random ratings to a new CSV file
    save_data(output_file_path, data_with_random_ratings)
    
    print("Random ratings have been successfully added to the dataset and saved to 'out.csv'.")

if __name__ == "__main__":
    main()
