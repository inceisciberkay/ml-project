#!/usr/bin/python3

# splits data into into fractured and nonfractured
import pandas as pd

input_csv_path = 'dataset.csv'
fracture_csv_path = 'fracture.csv'
nonfracture_csv_path = 'nonfracture.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(input_csv_path)

# Separate fractures and non-fractures
fractures = df[df['fracture_visible'] == 1]
nonfractures = df[df['fracture_visible'].isna()]

# Save the fractures and non-fractures DataFrames to new CSV files
fractures.to_csv(fracture_csv_path, index=False)
nonfractures.to_csv(nonfracture_csv_path, index=False)