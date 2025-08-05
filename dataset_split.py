# Data Collection and Data Preprocessing
# This script splits a dataset of soil images into Train, Validation, and Test sets.
import os
import shutil
import random
from pathlib import Path

# Set paths
source_dir = Path("MINI PROJECT")  # This is your original folder with soil categories
output_base = Path("BRANCH_MINELYTICS")  # Output root folder

# Create Train, Validation, and Test folders
splits = ['Train', 'Validation', 'Test']
for split in splits:
    for class_name in os.listdir(source_dir):
        class_dir = source_dir / class_name
        if class_dir.is_dir():
            os.makedirs(output_base / split / class_name, exist_ok=True)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Loop through each soil class and split
# Assuming each class folder contains images
for class_name in os.listdir(source_dir):
    class_dir = source_dir / class_name
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*.*"))  # All files in the class folder
    random.shuffle(images)

    total = len(images)
    train_count = int(train_ratio * total)
    val_count = int(val_ratio * total)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Copy files to their respective folders
    for img in train_images:
        shutil.copy(img, output_base / "Train" / class_name / img.name)
    for img in val_images:
        shutil.copy(img, output_base / "Validation" / class_name / img.name)
    for img in test_images:
        shutil.copy(img, output_base / "Test" / class_name / img.name)

print("Dataset successfully split into Train, Validation, and Test sets.")