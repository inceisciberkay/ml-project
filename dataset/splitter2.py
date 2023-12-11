#!/usr/bin/python3

import pandas as pd
import os
import shutil

dataset_csv_file_path = 'dataset.csv'
image_source_directory = '../all_images'

# Set the paths for positive and negative image folders
positive_folder = 'images2/fractured'
negative_folder = 'images2/nonfractured'

# Create folders if they don't exist
os.makedirs(positive_folder, exist_ok=True)
os.makedirs(negative_folder, exist_ok=True)

def separate_classes():
    df = pd.read_csv(dataset_csv_file_path)

    for _, row in df.iterrows():
        filename = row['filename']
        fracture_visible = row['fracture_visible']
        image_path = os.path.join(image_source_directory, filename + '.png')

        if os.path.isfile(image_path):
            destination_folder = positive_folder if fracture_visible == 1 else negative_folder

            shutil.copy(image_path, os.path.join(destination_folder, filename + '.png'))

def main():
    separate_classes()

main()