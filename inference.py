import PIL
import requests
import torch
from diffusers import  StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, PaintByExamplePipeline,AutoPipelineForText2Image
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import sys
from datetime import datetime
import os
from diffusers.utils import load_image
from transformers import CLIPImageProcessor

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/adi.tsach/diffusers/src/diffusers/pipelines/stable_diffusion/')
from pipeline_stable_diffusion_instruct_pix2pix_image import StableDiffusionInstructPix2PixImagePipeline
from io import BytesIO
model_id = "instruct-pix2pix-model" # <- replace this
# model_id = "runwayml/stablde-iffusion-v1-5"
pipe = StableDiffusionInstructPix2PixImagePipeline.from_pretrained(model_id, device_type="cuda", torch_dtype=torch.float16).to("cuda")
# pipe = StableDiffusionInstructPix2PixImagePipeline.from_pretrained(model_id,device_type="cuda", torch_dtype=torch.float16).to("cuda")
# pipe.feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")

generator = torch.Generator("cuda").manual_seed(0)
# pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
# pipe.set_ip_adapter_scale(0.7)
output_dir = "edited_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# url= "https://www.mansfieldtexas.gov/ImageRepository/Document?documentId=6501"
# url = "https://www.waco-texas.com/files/sharedassets/public/v/1/departments/parks-amp-recreation/images/parks-pictures/cameron-park/cameron-park.jpg?dimension=pageimage&w=480"
# url2 = "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg"
# url2 = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
# url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"
url="https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
url2 = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"
#url2 = "https://pbs.twimg.com/media/DlL5rKvW0AATU8n?format=jpg"
#url = "https://pooperscoopers.motivatedbrands.ca/wp-content/uploads/sites/2/2024/08/A-dog-in-a-red-bucket-that-is-used-for-pet-waste-bucket-services-offered-by-Motivated-Pooper-Scoopers.png"
# url2= "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQhVoYoLKufGfPnWI6UjkDT8O0h3qFjcLfFMQ&s"
# url2 = "https://i.natgeofe.com/k/0ed36c42-672a-425b-9e62-7cc946b98051/pig-fence_square.jpg"
#url2= "https://cdn.shopify.com/s/files/1/0016/1467/6056/products/vb-bucket-hat-2-1.jpg?v=1654732624"
#url ="https://www.flyovercanada.com/FlyOverCanada/media/FlyOver/Stories/2021/03/BL-Park-Pathways.jpg"
#url2 = "https://thumbs.dreamstime.com/b/american-stafford-cropped-ears-24481104.jpg"

# def download_image_for_en(url):
#     response = requests.get(url)
#     image = Image.open(BytesIO(response.content))
#     return image

# def preprocess_image(image, size=(224, 224)):
#     transform = transforms.Compose([
#         transforms.Resize(size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
#     return transform(image).unsqueeze(1)  # Add batch dimension

# def get_image_encoding(url, encoder):
#     image = download_image(url)
#     image_tensor = preprocess_image(image)
#     with torch.no_grad():
#         encoding = encoder(image_tensor)
#     return encoding

# Functions to download and process images
train_transforms = transforms.Compose(
    [
        transforms.transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),  # Random flip for augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalization
    ]
)

def download_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def preprocess_single_image(image, transforms):
    """Preprocesses a single image using the specified transformations."""
    # image = transforms(image)  # Apply transformations
    # return image.unsqueeze(0)  # Add batch dimension
    return image



# image = download_image(url)
# image2 = download_image(url2)
# image=preprocess_single_image(image, train_transforms)
# image2 = preprocess_single_image(image2, train_transforms)
# Download images
image = download_image(url)
image2 = download_image(url2)

# # Apply the feature extractor
# image = pipe.feature_extractor(images=image, return_tensors="pt").pixel_values.to(device='cuda:0')
# image2 = pipe.feature_extractor(images=image2, return_tensors="pt").pixel_values.to(device='cuda:0')

# Resize if necessary
# if image.shape[-1] != 224 or image.shape[-2] != 224:
#     image = torch.nn.functional.interpolate(image, size=(224, 224))
# if image2.shape[-1] != 224 or image2.shape[-2] != 224:
#     image2 = torch.nn.functional.interpolate(image2, size=(224, 224))


num_inference_steps = 120
image_guidance_scale = 1.3  #closer to og_image
guidance_scale = 2  # closer to ob_image
device = 'cuda:0'
pipe.scheduler.set_timesteps(num_inference_steps, device='cuda:0')
edited_image = pipe( 
   prompt="",
#    image=image.to(device=device),
#    ob_image =image2.to(device=device),
    image= image,
    ob_image = image2,
    # ip_adapter_image =image2,
   num_inference_steps=num_inference_steps,
   image_guidance_scale=image_guidance_scale,
   guidance_scale=guidance_scale,
#    generator=generator,
).images[0]


file_name = f"edited_image_{current_time}_gs-{guidance_scale}_igs-{image_guidance_scale}_nis_{num_inference_steps}.png"
file_path = os.path.join(output_dir, file_name)

# Save the image with the constructed path
edited_image.save(file_path)