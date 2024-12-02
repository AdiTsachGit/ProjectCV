import os
import shutil
import csv
from tqdm import tqdm

# Define main folder paths
main_folder = '/home/adi.tsach/diffusers/examples/instruct_pix2pix/new_dataset_2'
new_train_folder_base = '/home/adi.tsach/'
train_folder = os.path.join(new_train_folder_base, 'train')

# Subfolder paths
object_image_folder = os.path.join(train_folder, 'object_image')
target_image_folder = os.path.join(train_folder, 'target_image')
original_image_folder = os.path.join(train_folder, 'original_image')

# Create necessary folders
os.makedirs(object_image_folder, exist_ok=True)
os.makedirs(target_image_folder, exist_ok=True)
os.makedirs(original_image_folder, exist_ok=True)

# Initialize metadata list
metadata = []

# Base path for the metadata file format
base_path_for_metadata = "project_dataset_from_PIPE_train"

# Get subfolders
subfolders = [
    f for f in os.listdir(main_folder)
    if os.path.isdir(os.path.join(main_folder, f))
]

# Process subfolders with progress bar
for subfolder_name in tqdm(subfolders, desc="Processing subfolders"):
    subfolder_path = os.path.join(main_folder, subfolder_name)
    img_id = subfolder_name

    # Define file paths
    object_image_src = os.path.join(subfolder_path, f'object_image_{img_id}.jpg')
    target_image_src = os.path.join(subfolder_path, f'target_image_{img_id}.jpg')
    original_image_src = os.path.join(subfolder_path, f'original_image_{img_id}.jpg')

    # Define destination paths for copying
    object_image_dest = os.path.join(object_image_folder, f'{img_id}.jpg')
    target_image_dest = os.path.join(target_image_folder, f'{img_id}.jpg')
    original_image_dest = os.path.join(original_image_folder, f'{img_id}.jpg')

    # Log metadata with the desired format
    metadata.append({
        "img_id": img_id,
        "original_image": f"original_image/{img_id}.jpg" if os.path.exists(original_image_src) else "missing_file",
        "target_image": f"target_image/{img_id}.jpg" if os.path.exists(target_image_src) else "missing_file",
        "object_image": f"object_image_/{img_id}.jpg" if os.path.exists(object_image_src) else "missing_file"
    })

    # Copy files to their respective folders
    if os.path.exists(original_image_src):
        shutil.copy(original_image_src, original_image_dest)
    if os.path.exists(target_image_src):
        shutil.copy(target_image_src, target_image_dest)
    if os.path.exists(object_image_src):
        shutil.copy(object_image_src, object_image_dest)

# Write metadata to CSV if data exists
metadata_file = os.path.join(train_folder, 'metadata.csv')
if metadata:
    with open(metadata_file, 'w', newline='') as csvfile:
        fieldnames = ["img_id", "original_image", "target_image", "object_image"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    print("\nMetadata.csv successfully created.")
else:
    print("No metadata to write.")
