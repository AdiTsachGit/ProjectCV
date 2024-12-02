import torch
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModelWithProjection
from PIL import Image
import requests
from io import BytesIO

# Load the models and processor
vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Move models to device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
vision_model = vision_model.to(device)
text_model = text_model.to(device)

# 1. Load image from URL
image_url = "https://www.mansfieldtexas.gov/ImageRepository/Document?documentId=6501"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Preprocess image (CLIP expects a certain size and format)
inputs = processor(images=image, return_tensors="pt").to(device)

# 2. Prepare text inputs (ground truth labels or description)
text_inputs = ["dog in a park", "cat", "dog", "cat", "ant"]
tokenized_text = tokenizer(text_inputs, padding=True, return_tensors="pt").to(device)

# 3. Get image and text embeddings
with torch.no_grad():
    # Generate image embeddings
    image_embeddings = vision_model(**inputs).image_embeds  # Shape: [1, 1, 768]
    
    # Generate text embeddings
    text_outputs = text_model(**tokenized_text)
    text_embeddings = text_outputs.text_embeds  # Shape: [5, 77, 768]

print(f"Image embeddings shape: {image_embeddings.shape}")
print(f"Text embeddings shape: {text_embeddings.shape}")


# 5. Compute cosine similarity between image embeddings and text embeddings
# Since image_embeddings is of shape [1, 1, 768], we need to remove the extra dimensions
image_embeddings = image_embeddings.squeeze(1)  # Shape: [1, 768]

# Compute cosine similarity with both options
cosine_similarity_cls = torch.nn.functional.cosine_similarity(image_embeddings, text_embeddings)
cosine_similarity_mean = torch.nn.functional.cosine_similarity(image_embeddings, text_embeddings)

# Print the results
print(f"Cosine similarity using CLS token: {cosine_similarity_cls}")
print(f"Cosine similarity using mean of token embeddings: {cosine_similarity_mean}")

# Optionally: Choose a threshold to decide if the vectors are "similar"
similarity_threshold = 0.8  # You can adjust this based on your task
are_similar_cls = cosine_similarity_cls > similarity_threshold
are_similar_mean = cosine_similarity_mean > similarity_threshold

print(f"Are the image and text embeddings similar (CLS method)? {are_similar_cls}")
print(f"Are the image and text embeddings similar (Mean method)? {are_similar_mean}")
