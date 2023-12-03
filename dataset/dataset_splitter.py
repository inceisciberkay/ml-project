#!/usr/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

all_images_folder_path = "../all_images"
all_labels_folder_path = "../all_labels"

dataset_csv_file_path = './dataset.csv'
train_csv_file_path = './train.csv'
validation_csv_file_path = './validation.csv'
test_csv_file_path = './test.csv'

images_folder_path = './images'
train_images_folder_path = os.path.join(images_folder_path, 'train')
validation_images_folder_path = os.path.join(images_folder_path, 'validation') 
test_images_folder_path = os.path.join(images_folder_path, 'test')

labels_folder_path = './labels'
train_labels_folder_path = os.path.join(labels_folder_path, 'train')
validation_labels_folder_path = os.path.join(labels_folder_path, 'validation')
test_labels_folder_path = os.path.join(labels_folder_path, 'test')

label_names = ['nonfractured', 'fractured']

# Split the dataset.csv into train.csv, validation.csv, and test.csv
def split_csv():
    df = pd.read_csv(dataset_csv_file_path)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    validation_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)

    train_df.to_csv(train_csv_file_path, index=False)
    validation_df.to_csv(validation_csv_file_path, index=False)
    test_df.to_csv(test_csv_file_path, index=False)

def create_subfolders():
    # Read the CSV files
    train_df = pd.read_csv(train_csv_file_path)
    validation_df = pd.read_csv(validation_csv_file_path)
    test_df = pd.read_csv(test_csv_file_path)

    # recreate images and labels directories (.csv file contents might be changed)
    if os.path.exists(images_folder_path):
        shutil.rmtree(images_folder_path)
    if os.path.exists(labels_folder_path):
        shutil.rmtree(labels_folder_path)

    os.makedirs(os.path.join(train_images_folder_path, label_names[0]), exist_ok=True)
    os.makedirs(os.path.join(train_images_folder_path, label_names[1]), exist_ok=True)
    os.makedirs(os.path.join(validation_images_folder_path, label_names[0]), exist_ok=True)
    os.makedirs(os.path.join(validation_images_folder_path, label_names[1]), exist_ok=True)
    os.makedirs(os.path.join(test_images_folder_path, label_names[0]), exist_ok=True)
    os.makedirs(os.path.join(test_images_folder_path, label_names[1]), exist_ok=True)

    os.makedirs(os.path.join(train_labels_folder_path, label_names[0]), exist_ok=True)
    os.makedirs(os.path.join(train_labels_folder_path, label_names[1]), exist_ok=True)
    os.makedirs(os.path.join(validation_labels_folder_path, label_names[0]), exist_ok=True)
    os.makedirs(os.path.join(validation_labels_folder_path, label_names[1]), exist_ok=True)
    os.makedirs(os.path.join(test_labels_folder_path, label_names[0]), exist_ok=True)
    os.makedirs(os.path.join(test_labels_folder_path, label_names[1]), exist_ok=True)

    def copy_images_and_labels(df, destination_image_split_path, destination_label_split_path):
        for _, row in df.iterrows():
            label = label_names[int(row['fracture_visible'])]
            image_path = row['filename'] + '.png'
            label_path = row['filename'] + '.txt'

            source_image_path = os.path.join(all_images_folder_path, image_path)
            source_label_path = os.path.join(all_labels_folder_path, label_path)

            destination_image_path = os.path.join(destination_image_split_path, label)
            destination_label_path = os.path.join(destination_label_split_path, label)

            shutil.copy(source_image_path, destination_image_path)
            shutil.copy(source_label_path, destination_label_path)

    copy_images_and_labels(train_df, train_images_folder_path, train_labels_folder_path)
    copy_images_and_labels(validation_df, validation_images_folder_path, validation_labels_folder_path)
    copy_images_and_labels(test_df, test_images_folder_path, test_labels_folder_path)

    # print("Image directories organized for TensorFlow.")

def main():
    split_csv()
    create_subfolders()

main()