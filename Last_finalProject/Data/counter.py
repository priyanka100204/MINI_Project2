# This script is for counting the dataset

import os
import csv

def count_rows_in_csv_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    
    # Dictionary to store the row count for each file
    row_counts = {}
    
    # Iterate over each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader) - 1  # Subtract 1 for header row
            row_counts[csv_file] = row_count
    
    return row_counts

def get_count(folder_path):
    # Get the row counts
    row_counts = count_rows_in_csv_files(folder_path)
    total_data = sum(row_counts.values())
    # Print the row counts
    for csv_file, row_count in row_counts.items():
        print(f'{csv_file}: {row_count} rows')
    print(f'Total data: {total_data} rows\n')



# Specify the folder path
Bitumin = 'Data/Bitumin'
Block = 'Data/Block'
Concrete = 'Data/Concrete'
Kanker = 'Data/Kanker'


# call get_count function for each folder
get_count(Bitumin)
get_count(Block)
get_count(Concrete)
get_count(Kanker)
