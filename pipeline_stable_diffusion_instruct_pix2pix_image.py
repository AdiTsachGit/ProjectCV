# # Copyright 2023 The InstructPix2Pix Authors and The HuggingFace Team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.


from diffusers.utils import (
    PIL_INTERPOLATION,

)

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.paint_by_example import PaintByExampleImageEncoder

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPVisionModel
import torch.optim as optim

import os

# Define the `EmbeddingOptimizer` class, as per the provided code.
class EmbeddingOptimizer(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, input_seq_len=257, output_seq_len=77):
        super(EmbeddingOptimizer, self).__init__()
        self.projection_layer = torch.nn.Linear(input_dim, output_dim)
        self.attention_pooling = torch.nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        self.mlp_fc1 = torch.nn.Linear(output_dim, output_dim)
        self.mlp_relu = torch.nn.ReLU()
        self.mlp_fc2 = torch.nn.Linear(output_dim, output_dim)
        self.output_seq_len = output_seq_len
        self.input_seq_len = input_seq_len

    def forward(self, x):
        # Step 1: Project input embeddings from 1024 to 768 dimensions
        x = self.projection_layer(x)  # Shape: [batch_size, 257, 768]

        # Step 2: Attention pooling to reduce sequence length from 257 to 77
        # Create a query tensor for pooling: [batch_size, 77, 768]
        query = torch.randn(x.size(0), self.output_seq_len, x.size(-1), device=x.device, dtype=x.dtype)
        x, _ = self.attention_pooling(query, x, x)

        # Step 3: Apply the MLP block for further refinement
        x = self.mlp_fc1(x)
        x = self.mlp_relu(x)
        x = self.mlp_fc2(x)

        return x

embedding_optimizer = EmbeddingOptimizer(
    input_dim=1024,    # The input dimension (from your image encoder)
    output_dim=768,    # The desired output dimension (for the UNet cross-attention)
    input_seq_len=257, # The input sequence length (from your image encoder)
    output_seq_len=77  # The target sequence length (for Stable Diffusion's context length)
)

class AttentionPooling(torch.nn.Module):
    def __init__(self, input_dim, output_tokens):
        super().__init__()
        self.query = torch.nn.Linear(input_dim, input_dim)
        self.output_tokens = output_tokens

    def forward(self, hidden_states):
        # Compute attention scores
        query = self.query(hidden_states[:, 0:1, :])  # Use CLS token as the query
        attention_scores = torch.matmul(query, hidden_states.transpose(-1, -2)) / (hidden_states.size(-1) ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply attention pooling
        pooled_output = torch.matmul(attention_weights, hidden_states)  # Shape: [batch_size, 1, input_dim]
        pooled_output = pooled_output.expand(-1, self.output_tokens, -1)  # Expand to match sequence length
        return pooled_output


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")



def prepare_image(image):
    if isinstance(image, torch.Tensor):


        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        assert image.ndim == 4 


        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            image = image / 127.5 - 1.0  # Normalize image to [-1, 1] range


        # Image as float32 aya check
        image = image.to(dtype=torch.float32)

    else:
        if isinstance(image, PIL.Image.Image):
            image = [image]

        image = np.concatenate([np.array(i.convert("RGB"))[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image

class StableDiffusionInstructPix2PixImagePipeline(StableDiffusionInstructPix2PixPipeline):


    
    def prepare_image_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"image has to be of type torch.Tensor, PIL.Image.Image or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
        else:
            image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (prompt), but only {image_latents.shape[0]} initial"
                " images (image). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate image of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    

    def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        # Preprocess the image if not already a tensor
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
            print("did it")

        # Resize if necessary
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            image = torch.nn.functional.interpolate(image, size=(224, 224))

        # Send image to the appropriate device and dtype
        image = image.to(device=device, dtype=dtype)
        self.image_encoder = self.image_encoder.to(device=device, dtype=dtype)

        # Get the image embeddings (hidden states)
        image_embeddings = self.image_encoder(image).last_hidden_state  # Shape: [batch_size, 768]

        # Apply the Embedding Optimizer
        optimizer_state = torch.load("/home/adi.tsach/diffusers/examples/instruct_pix2pix/instruct-pix2pix-model/emb_opt.pt")
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = torch.nan_to_num(value, nan=0.0)
        embedding_optimizer.load_state_dict(optimizer_state)
        embedding_optimizer.to(device=device, dtype=dtype)
        image_embeddings = embedding_optimizer(image_embeddings)

        # Apply dropout for classifier-free guidance
        dropout_prob = 0.05  # Use the same probability as in training
        image_embeddings = torch.nn.functional.dropout(image_embeddings, p=dropout_prob, training=False)
        num_zero_vectors = (torch.norm(image_embeddings, p=2, dim=-1) == 0).sum()
        print(f"Number of zero embeddings: {num_zero_vectors}")

        # For classifier-free guidance, prepare unconditional embeddings
        if do_classifier_free_guidance:
            uncond_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([image_embeddings, uncond_image_embeddings, uncond_image_embeddings], dim=0)

        return image_embeddings  # Shape: [batch_size * 3, 768]




    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        ob_image = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (str or List[str], optional):
                The prompt or prompts to guide the image generation. If not defined, one has to pass prompt_embeds.
                instead.
            image (PIL.Image.Image):
                Image, or tensor representing an image batch which will be repainted according to prompt.
            num_inference_steps (int, optional, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (float, optional, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                guidance_scale is defined as w of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text prompt,
                usually at the expense of lower image quality. This pipeline requires a value of at least 1.
            image_guidance_scale (float, optional, defaults to 1.5):
                Image guidance scale is to push the generated image towards the inital image image. Image guidance
                scale is enabled by setting image_guidance_scale > 1. Higher image guidance scale encourages to
                generate images that are closely linked to the source image image, usually at the expense of lower
                image quality. This pipeline requires a value of at least 1.
            negative_prompt (str or List[str], optional):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                negative_prompt_embeds. instead. Ignored when not using guidance (i.e., ignored if guidance_scale
                is less than 1).
            num_images_per_prompt (int, optional, defaults to 1):
                The number of images to generate per prompt.
            eta (float, optional, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [schedulers.DDIMScheduler], will be ignored for others.
            generator (torch.Generator, optional):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (torch.FloatTensor, optional):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random generator.
            prompt_embeds (torch.FloatTensor, optional):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, e.g. prompt weighting. If not
                provided, text embeddings will be generated from prompt input argument.
            negative_prompt_embeds (torch.FloatTensor, optional):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, e.g. prompt
                weighting. If not provided, negative_prompt_embeds will be generated from negative_prompt input
                argument.
            output_type (str, optional, defaults to "pil"):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): PIL.Image.Image or np.array.
            return_dict (bool, optional, defaults to True):
                Whether or not to return a [~pipelines.stable_diffusion.StableDiffusionPipelineOutput] instead of a
                plain tuple.
            callback (Callable, optional):
                A function that will be called every callback_steps steps during inference. The function will be
                called with the following arguments: callback(step: int, timestep: int, latents: torch.FloatTensor).
            callback_steps (int, optional, defaults to 1):
                The frequency at which the callback function will be called. If not specified, the callback will be
                called at every step.

        Examples:

        py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInstructPix2PixPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

        >>> image = download_image(img_url).resize((512, 512))

        >>> pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        ...     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "make the mountains snowy"
        >>> image = pipe(prompt=prompt, image=image).images[0]
        

        Returns:
            [~pipelines.stable_diffusion.StableDiffusionPipelineOutput] or tuple:
            [~pipelines.stable_diffusion.StableDiffusionPipelineOutput] if return_dict is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the safety_checker.
        """
        # 0. Check inputs
        self.check_inputs(prompt, callback_steps)

        if image is None:
            raise ValueError("image input cannot be undefined.")
        
        # 1. Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt) aya
        batch_size = 1
        self.attention_pooling = AttentionPooling(input_dim=1024, output_tokens=77).to("cuda:0")

        device = self._execution_device
        # here guidance_scale is defined analog to the guidance weight w of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . guidance_scale = 1
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0 
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if not isinstance(ob_image, torch.Tensor):
            ob_image = self.feature_extractor(images=ob_image, return_tensors="pt").pixel_values
        if ob_image.shape[-1] != 224 or ob_image.shape[-2] != 224:
            ob_image = torch.nn.functional.interpolate(ob_image, size=(224, 224))

# Encode the preprocessed image
        image_embeddings = self._encode_image(ob_image, device, num_images_per_prompt, do_classifier_free_guidance)
        cosine_similarity = torch.cosine_similarity(image_embeddings, prompt_embeds, dim=-1)  # Shape: (1, 77)
        similarity_value = torch.mean(cosine_similarity[0])
        max_sim = torch.max(cosine_similarity[0])

        print("mean sim:", similarity_value)
        print("max sim: ", max_sim)
# Calculate the average cosine similarity across all tokens (77)
        average_cosine_similarity = cosine_similarity.mean().item()

        print(average_cosine_similarity)

        og_image = prepare_image(image)
        height, width = og_image.shape[-2:]

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        #5. Prepare Image latents
        image_latents = self.prepare_image_latents(
            og_image,
            batch_size,
            num_images_per_prompt,
            image_embeddings.dtype,
            device,
            do_classifier_free_guidance,
            generator,
        )
        

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of pipeline.unet: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received num_channels_latents: {num_channels_latents} +"
                f" num_channels_image: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " pipeline.unet or your image input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents 


                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                scaled_latent_model_input = torch.cat([scaled_latent_model_input,  image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(scaled_latent_model_input, t, encoder_hidden_states=image_embeddings).sample
                
                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == t).nonzero().item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - latents) / (-sigma)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 10. Post-processing
        image = self.decode_latents(latents)

        # 11. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, image_embeddings.dtype)

        # 12. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)
        

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)