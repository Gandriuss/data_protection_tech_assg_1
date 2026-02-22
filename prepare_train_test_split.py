import pandas as pd
import csv
import os
import shutil
from pathlib import Path

file_path = "identity_CelebA_jpg.txt"
train_to_test_split_ratio = 0.7


collector = []
f = open(file_path, "r")
for line in f.readlines():
    img_name, iden = line.strip().split(' ')
    collector.append({"name": img_name, "label": iden})

labelsdf = pd.DataFrame(collector)


# 1. Get the counts (as we did before)
counts = labelsdf['label'].value_counts()

# 2. Identify labels that have at least 6 occurrences
# This creates a list or Index of labels to keep
keep_labels = counts[counts >= 10].index

# 3. Filter the original dataframe
# .isin() is the most efficient way to check against a list
filtered_df = labelsdf[labelsdf['label'].isin(keep_labels)]

# Optional: Verify the results
print(f"Original rows: {len(labelsdf)}")
print(f"Filtered rows: {len(filtered_df)}")
print(f"Unique labels remaining: {filtered_df['label'].nunique()}")


# Get the unique labels in their current order
unique_labels = filtered_df['label'].unique()

# label_split_index = int(round(unique_labels.shape[0]*train_to_test_split_ratio,0))

# # Slice the first 4,000 unique label IDs
# train_label_ids = unique_labels[:label_split_index]

# # Slice the next 4,000 unique label IDs
# test_label_ids = unique_labels[label_split_index:]

# Slice the first 4,000 unique label IDs
train_label_ids = unique_labels[:1000]

# Slice the next 4,000 unique label IDs
test_label_ids = unique_labels[1000:2000]


# Pull all rows where the label is in our "train" list
train_df = filtered_df[filtered_df['label'].isin(train_label_ids)]

# Pull all rows where the label is in our "test" list
test_df = filtered_df[filtered_df['label'].isin(test_label_ids)]

print(f"Train DF has {train_df['label'].nunique()} labels and {len(train_df)} total images.")
print(f"Test DF has {test_df['label'].nunique()} labels and {len(test_df)} total images.")


# Export Train
train_df.to_csv('data/trainset.txt', sep=' ', header=False, index=False)

# Export Test
test_df.to_csv('data/testset.txt', sep=' ', header=False, index=False)


# 1. Define your paths
source_folder = "./data/img_align_celeba/img_align_celeba"
target_folder = "./data/img_align_celeba_filtered"

# 2. Create a set for O(1) lightning-fast lookups
pictures_to_keep = set(train_df['name']).union(set(test_df['name']))

print(f"Total images in the 'keep' list: {len(pictures_to_keep)}")

# 3. Iterate through the folder and delete what we don't need
deleted_count = 0
kept_count = 0

# Get list of all files in the directory
all_files = os.listdir(source_folder)

for filename in all_files:
    file_path = os.path.join(source_folder, filename)
    
    # If the file is NOT in our keep set, delete it
    if filename not in pictures_to_keep:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
    else:
        kept_count += 1

print(f"Cleanup complete!")
print(f"Deleted: {deleted_count} unwanted images.")
print(f"Kept: {kept_count} images in {source_folder}.")