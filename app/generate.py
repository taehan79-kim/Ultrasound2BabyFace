import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
    feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()
    return depth_estimator, feature_extractor, pipe

def get_depth_map(image, depth_estimator, feature_extractor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(device):
        depth_map = depth_estimator(image).predicted_depth
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    return Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))

def generate_image(depth_estimator, feature_extractor, pipe, input_image, prompt = None, negative_prompt = None, seed = None, strength = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # depth_estimator, feature_extractor, pipe = load_models()
    image = input_image.resize((1024, 1024))
    depth_image = get_depth_map(image, depth_estimator, feature_extractor)
    if prompt is None:
        prompt = "A professional portrait of an adorable Asian baby sleeping, peaceful, serene, soothing, closed eyes, eyelashes, 8k photo"
    if negative_prompt is None:
        negative_prompt = "poorly rendered face, poorly drawn face, poor facial details, blurry image, bad anatomy"
    if seed is None:
        seed = 123456772
    if strength is None:
        strength = 0.70
    generator = torch.Generator(device=device).manual_seed(seed)
    images = pipe(
        prompt,
        image=image,
        control_image=depth_image,
        strength=strength,
        num_inference_steps=50,
        controlnet_conditioning_scale=0.5,
        generator=generator,
        negative_prompt=negative_prompt,
    ).images
    return images[0]