import numpy as np
from PIL import Image
import os
import pandas as pd
from datasets import load_dataset
import io
import torch

# Configuration
dir_name = 'new_dataset_2'
os.makedirs(dir_name, exist_ok=True)
sample_size = 120000
batch_size = 1000
metadata_csv_path = os.path.join(dir_name, "metadata.csv")

# Load datasets
masks_dataset = load_dataset('paint-by-inpaint/PIPE_Masks', split='train')
images_dataset = load_dataset('paint-by-inpaint/PIPE', split='train')

selected_columns = ['img_id', 'target_img', 'source_img']

# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions
def ensure_pil_image(data):
    """Ensure the input is a PIL Image, converting if necessary."""
    if isinstance(data, Image.Image):
        return data
    elif isinstance(data, (bytes, bytearray)):
        return Image.open(io.BytesIO(data))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def mask_to_bbox(mask):
    mask_tensor = torch.tensor(mask, device=device)
    rows = torch.any(mask_tensor, dim=1)
    cols = torch.any(mask_tensor, dim=0)

    if not torch.any(rows) or not torch.any(cols):
        return None

    y_min, y_max = torch.where(rows)[0][[0, -1]].cpu().numpy()
    x_min, x_max = torch.where(cols)[0][[0, -1]].cpu().numpy()

    return x_min, y_min, x_max - x_min, y_max - y_min

def normalize_bbox(mask_shape, bbox):
    x_min, y_min, width, height = bbox
    mask_height, mask_width = mask_shape

    x_scale = 512 / mask_width
    y_scale = 512 / mask_height

    return int(x_min * x_scale), int(y_min * y_scale), int(width * x_scale), int(height * y_scale)

def crop_image_by_bbox(image, bbox):
    x_min, y_min, width, height = bbox
    return image.crop((x_min, y_min, x_min + width, y_min + height))

def save_cropped_image(image_id, bbox, target_image_data, original_image_data, metadata):
    try:
        # Ensure images are PIL Images
        target_image = ensure_pil_image(target_image_data)
        original_image = ensure_pil_image(original_image_data)

        # Crop target image using the bounding box
        cropped_image = crop_image_by_bbox(target_image, bbox)

        # Save images
        image_dir = os.path.join(dir_name, str(image_id))
        os.makedirs(image_dir, exist_ok=True)

        object_image_path = os.path.join(image_dir, f"object_image_{image_id}.jpg")
        target_image_path = os.path.join(image_dir, f"target_image_{image_id}.jpg")
        original_image_path = os.path.join(image_dir, f"original_image_{image_id}.jpg")

        cropped_image.save(object_image_path)
        target_image.save(target_image_path)
        original_image.save(original_image_path)

        # Append metadata
        metadata.append({
            'id': str(image_id),
            'object_image': object_image_path,
            'target_image': target_image_path,
            'original_image': original_image_path
        })

        print(f"Cropped images saved for {image_id}.")
    except Exception as e:
        print(f"Error processing {image_id}: {e}")

# Processing loop
metadata = []
for i in range(0, sample_size, batch_size):
    print(f"Processing batch {i} to {i + batch_size}...")

    # Load batch
    batch = images_dataset.select(range(i, min(i + batch_size, sample_size)))
    batch = batch.remove_columns([col for col in batch.column_names if col not in selected_columns])
    batch_df = pd.DataFrame(batch)

    # Process each mask and associated image
    for idx, mask_data in enumerate(masks_dataset.select(range(i, min(i + batch_size, sample_size)))):
        try:
            image_id = mask_data['img_id']
            mask = ensure_pil_image(mask_data['mask']).convert('L')
            mask_np = np.array(mask)

            # Get bounding box
            bbox = mask_to_bbox(mask_np)
            if bbox:
                bbox = normalize_bbox(mask_np.shape, bbox)

                # Fetch image data
                row = batch_df.loc[batch_df['img_id'] == image_id]
                if row.empty:
                    print(f"No matching image found for {image_id}.")
                    continue

                target_image_data = row.iloc[0]['target_img']
                original_image_data = row.iloc[0]['source_img']

                # Save cropped images
                save_cropped_image(image_id, bbox, target_image_data, original_image_data, metadata)
        except Exception as e:
            print(f"Skipping mask {mask_data['img_id']} due to error: {e}")

    # Write metadata incrementally
    if metadata:
        pd.DataFrame(metadata).to_csv(metadata_csv_path, mode='a', header=not os.path.exists(metadata_csv_path), index=False)
        metadata.clear()

print("Processing complete.")
