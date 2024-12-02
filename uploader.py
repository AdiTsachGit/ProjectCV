import os
import numpy as np
import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset

# Set your directory name
dir_name = './project_dataset_from_PIPE_train'

# print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_name}")
import os
import pandas as pd
from datasets import Dataset, Features, Value, Image

# Step 1: Define the base directory where your dataset is located
base_dir = "/home/adi.tsach/train"  # Replace with your base directory

# Step 2: Load your updated CSV file with image paths
csv_file = os.path.join(base_dir, "metadata-66.csv")
df = pd.read_csv(csv_file)
# Update the paths in the DataFrame to include the full path
df['original_image'] = df['original_image'].apply(lambda x: os.path.join(base_dir, x))
df['target_image'] = df['target_image'].apply(lambda x: os.path.join(base_dir, x))
df['object_image'] = df['object_image'].apply(lambda x: os.path.join(base_dir, x))

# Step 3: Define the dataset features, marking image columns with Image()
features = Features({
    'img_id': Value('string'),
    'original_image': Image(),  # Third image
    'target_image': Image(),    # Second image
    'object_image': Image()    # First image

})

# Step 4: Create the dataset from the DataFrame
dataset = Dataset.from_pandas(df, features=features)

# Step 5: Remove any unnecessary columns
if "_index_level_0_" in dataset.column_names:
    dataset = dataset.remove_columns("_index_level_0_")

# Step 6: Push the dataset to Hugging Face Hub
dataset.push_to_hub("ADT1999/project_from_PIPE-bug")  # Replace with your username and dataset name

print("Dataset successfully uploaded!")