import pandas as pd
import csv
import os
import shutil
from pathlib import Path
import numpy as np

file_path = "identity_CelebA_jpg.txt"
train_to_test_split_ratio = 0.9

# ---- Tunables ----
# N_VICTIM_LABELS = 1500
# N_ATTACK_LABELS = 1000
N_VICTIM_LABELS = 1000
N_ATTACK_LABELS = 500
RANDOM_STATE = 42


def stratified_split_by_label(df: pd.DataFrame, label_col: str, train_ratio: float, random_state: int):
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    rng = np.random.RandomState(random_state)
    train_parts = []
    test_parts = []
    for _, g in df.groupby(label_col):
        g = g.sample(frac=1.0, random_state=int(rng.randint(0, 2**31 - 1)))
        n = len(g)
        n_train = int(round(n * train_ratio))
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
        train_parts.append(g.iloc[:n_train])
        test_parts.append(g.iloc[n_train:])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


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

# Victim: pick a label set, then split images (stratified) so train/test share labels
victim_label_ids = unique_labels[:N_VICTIM_LABELS]
victim_df = filtered_df[filtered_df['label'].isin(victim_label_ids)].copy()
train_df, test_df = stratified_split_by_label(
    victim_df,
    label_col='label',
    train_ratio=train_to_test_split_ratio,
    random_state=RANDOM_STATE,
)

# Attacker labels
attack_label_ids = unique_labels[N_VICTIM_LABELS:N_VICTIM_LABELS + N_ATTACK_LABELS]
attacking_train_df = filtered_df[filtered_df['label'].isin(attack_label_ids)].copy()

print(f"Victim labels: {pd.Series(victim_label_ids).nunique()} | victim images: {len(victim_df)}")
print(f"Train DF has {train_df['label'].nunique()} labels and {len(train_df)} total images.")
print(f"Test  DF has {test_df['label'].nunique()} labels and {len(test_df)} total images.")
print(f"Attack DF has {attacking_train_df['label'].nunique()} labels and {len(attacking_train_df)} total images.")

os.makedirs('data', exist_ok=True)

# Export victim train/test
train_df.to_csv('data/trainset.txt', sep=' ', header=False, index=False)
test_df.to_csv('data/testset.txt', sep=' ', header=False, index=False)

# Export attacker and keep-only
attacking_train_df.to_csv('data/ganset.txt', sep=' ', header=False, index=False)


# 1. Define source path
source_folder = "./data/img_align_celeba/img_align_celeba"

# 2. Create a set for O(1) lightning-fast lookups
pictures_to_keep = (
    set(train_df['name'])
    .union(set(test_df['name']))
    .union(set(attacking_train_df['name']))
)

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