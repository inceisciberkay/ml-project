#!/usr/bin/python3

import os
import shutil
import pandas

all_images_path = 'all_images'
all_labels_path = 'all_labels'

output_root_path = 'split'

split_csv_paths = ['fracture.csv', 'nonfracture.csv']
csv_delimiter = ',' # CSV column delimiter

move_files = False # Move files to destination, otherwise copy the files 

if not os.path.exists(all_images_path) or not os.path.exists(all_labels_path) or not all(os.path.isfile(csv_path) for csv_path in split_csv_paths):
    print('ERROR (Directory/file issues): Please check paths.')
    exit()

# Fill dictionary from columns
folder_set_map = {}
for csv_path in split_csv_paths:
    folder_name = csv_path.split('.')[0]
    folder_set_map[folder_name] = set(
        pandas.read_csv(os.path.normpath(csv_path), dtype=str, sep=csv_delimiter, 
        usecols=['filestem'])['filestem'])

# Create output directory structure
for folder_name, split_set in folder_set_map.items():
    split_output_path = os.path.join(output_root_path, folder_name)
    images_output_path = os.path.join(split_output_path, 'images')
    labels_output_path = os.path.join(split_output_path, 'labels')

    os.makedirs(images_output_path, exist_ok=True)
    os.makedirs(labels_output_path, exist_ok=True)

    for root, _, files in os.walk(all_images_path):
        for file in files:
            if os.path.splitext(file)[0] in split_set:
                source_image_path = os.path.join(root, file)
                destination_image_path = os.path.join(images_output_path, file)

                source_label_path = os.path.join(all_labels_path, f"{os.path.splitext(file)[0]}.txt")
                destination_label_path = os.path.join(labels_output_path, f"{os.path.splitext(file)[0]}.txt")

                if move_files:
                    shutil.move(source_image_path, destination_image_path)
                    shutil.move(source_label_path, destination_label_path)
                else:
                    shutil.copy2(source_image_path, destination_image_path)
                    shutil.copy2(source_label_path, destination_label_path)

print('Split operation completed.')