import os
import sys
from datetime import datetime
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

# Insert path for custom pipeline if needed
sys.path.insert(1, '/home/adi.tsach/diffusers/src/diffusers/pipelines/stable_diffusion/')
from pipeline_stable_diffusion_instruct_pix2pix_image import StableDiffusionInstructPix2PixImagePipeline

# Define model ID and load pipeline
model_id = "ADT1999/instruct-pix2pix-model-trial2"
pipe = StableDiffusionInstructPix2PixImagePipeline.from_pretrained(
    model_id,
    device_type="cuda",
    torch_dtype=torch.float16
).to("cuda")

# Set up the random generator for reproducibility
generator = torch.Generator("cuda").manual_seed(0)

# Output directory setup
output_dir = "edited_images"
os.makedirs(output_dir, exist_ok=True)

# Get the current date and time for the output file name
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Function to load an image from a local path
def load_image_from_path(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

#c14a08db41026f5f
# Image paths
img_id = "ae8dc7a50ef2b1d5"
local_image_path1 = f"/home/adi.tsach/diffusers/examples/instruct_pix2pix/new_dataset_2/{img_id}/original_image_{img_id}.jpg"
local_image_path2 = f"/home/adi.tsach/diffusers/examples/instruct_pix2pix/new_dataset_2/{img_id}/object_image_{img_id}.jpg"

# Load images
image = load_image_from_path(local_image_path1)
object_image = load_image_from_path(local_image_path2)

# Set inference parameters
num_inference_steps = 66
image_guidance_scale = 1.2
guidance_scale = 3

# Run the pipeline``
edited_image = pipe(
    prompt ="airplane",
    image=image,
    ob_image=object_image,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    # generator=generator
).images[0]

# Save the edited image
file_name = f"edited_image_{current_time}_gs-{guidance_scale}_igs-{image_guidance_scale}_nis_{num_inference_steps}_id_{img_id}.png"
file_path = os.path.join(output_dir, file_name)
edited_image.save(file_path)

print(f"Edited image saved at: {file_path}")
