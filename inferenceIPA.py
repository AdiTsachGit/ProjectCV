import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, PaintByExamplePipeline, AutoPipelineForText2Image
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO
from datetime import datetime
import os

# Load the required pipeline
from diffusers.utils import load_image

# Ensure you're using the correct Pix2Pix pipeline that supports IP Adapter
model_id = "instruct-pix2pix-model"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Ensure IP Adapter weights are loaded
if pipe.image_encoder is not None:
    print("Loading IP Adapter weights...")
    ip_adapter_weights_path = "./ip-adapter_sd15_light.bin"
    pipe.image_encoder.load_state_dict(torch.load(ip_adapter_weights_path, map_location="cuda"), strict=False)
    print("IP Adapter weights loaded successfully.")
else:
    print("Warning: IP Adapter is not initialized in the pipeline.")


# Prepare output directory
output_dir = "edited_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Download and process images
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

url = "https://pooperscoopers.motivatedbrands.ca/wp-content/uploads/sites/2/2024/08/A-dog-in-a-red-bucket-that-is-used-for-pet-waste-bucket-services-offered-by-Motivated-Pooper-Scoopers.png"
url2 = "https://pbs.twimg.com/media/DlL5rKvW0AATU8n?format=jpg"
#url2 = "https://pooperscoopers.motivatedbrands.ca/wp-content/uploads/sites/2/2024/08/A-dog-in-a-red-bucket-that-is-used-for-pet-waste-bucket-services-offered-by-Motivated-Pooper-Scoopers.png"

image = download_image(url)
image2 = download_image(url2)

# Set inference parameters
num_inference_steps = 80
image_guidance_scale = 1.9 # Adjust for image-based guidance
guidance_scale = 20.5  # Set to 1 to disable text prompt influence

# Perform inference using the ip_adapter_image argument
edited_image = pipe(
    prompt="add a zebra",
    image=image,
    ip_adapter_image=image2,  # Provide the example image for IP Adapter guidance
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,  # Text guidance scale set to 1 (classifier-free guidance)
).images[0]

# Save the output image
file_name = f"edited_image_{current_time}_gs-{guidance_scale}_igs-{image_guidance_scale}_{num_inference_steps}.png"
file_path = os.path.join(output_dir, file_name)
edited_image.save(file_path)

print(f"Edited image saved at: {file_path}")
