'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import concurrent.futures
import logging
import traceback
import sys

import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure CUDA is available
if not torch.cuda.is_available():
    logger.error("CUDA is not available. GPU is required for this worker.")
    sys.exit(1)

torch.cuda.empty_cache()

class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        try:
            model_path = "/FLUX.1-schnell"
            logger.info(f"Loading base model from {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path {model_path} does not exist")

            base_pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            base_pipe = base_pipe.to("cuda", silence_dtype_warnings=True)
            base_pipe.enable_xformers_memory_efficient_attention()
            logger.info("Base model loaded successfully")
            return base_pipe
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise

    def load_refiner(self):
        try:
            model_path = "/FLUX.1-schnell"
            logger.info("Loading refiner model")
            refiner_pipe = FluxImg2ImgPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            refiner_pipe = refiner_pipe.to("cuda", silence_dtype_warnings=True)
            refiner_pipe.enable_xformers_memory_efficient_attention()
            logger.info("Refiner model loaded successfully")
            return refiner_pipe
        except Exception as e:
            logger.error(f"Error loading refiner model: {str(e)}")
            raise

    def load_models(self):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_base = executor.submit(self.load_base)
                future_refiner = executor.submit(self.load_refiner)

                # Wait for both models to load with timeout
                self.base = future_base.result(timeout=300)  # 5 minute timeout
                self.refiner = future_refiner.result(timeout=300)

            logger.info("All models loaded successfully")
        except concurrent.futures.TimeoutError:
            logger.error("Timeout while loading models")
            raise
        except Exception as e:
            logger.error(f"Error in load_models: {str(e)}")
            raise

def _save_and_upload_images(images, job_id):
    try:
        output_dir = f"/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        image_urls = []

        for index, image in enumerate(images):
            image_path = os.path.join(output_dir, f"{index}.png")

            try:
                image.save(image_path)
                logger.info(f"Saved image to {image_path}")

                if os.environ.get('BUCKET_ENDPOINT_URL', False):
                    image_url = rp_upload.upload_image(job_id, image_path)
                    logger.info(f"Uploaded image {index} to bucket")
                    image_urls.append(image_url)
                else:
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(
                            image_file.read()).decode("utf-8")
                        image_urls.append(f"data:image/png;base64,{image_data}")
                        logger.info(f"Encoded image {index} as base64")

            except Exception as e:
                logger.error(f"Error processing image {index}: {str(e)}")
                raise

        rp_cleanup.clean([output_dir])
        return image_urls
    except Exception as e:
        logger.error(f"Error in save_and_upload_images: {str(e)}")
        raise

def make_scheduler(name, config):
    scheduler_map = {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }
    scheduler = scheduler_map.get(name)
    if scheduler is None:
        logger.warning(f"Unknown scheduler {name}, falling back to DDIM")
        scheduler = DDIMScheduler.from_config(config)
    return scheduler

MODELS = ModelHandler()

@torch.inference_mode()
def generate_image(job):
    try:
        job_input = job["input"]
        logger.info(f"Processing job with input: {job_input}")

        # Input validation
        validated_input = validate(job_input, INPUT_SCHEMA)
        if 'errors' in validated_input:
            logger.error(f"Input validation failed: {validated_input['errors']}")
            return {"error": validated_input['errors']}

        job_input = validated_input['validated_input']
        starting_image = job_input['image_url']

        if job_input['seed'] is None:
            job_input['seed'] = int.from_bytes(os.urandom(2), "big")
            logger.info(f"Generated random seed: {job_input['seed']}")

        generator = torch.Generator("cuda").manual_seed(job_input['seed'])

        MODELS.base.scheduler = make_scheduler(
            job_input['scheduler'], MODELS.base.scheduler.config)

        if starting_image:
            logger.info("Processing with starting image")
            init_image = load_image(starting_image).convert("RGB")
            output = MODELS.refiner(
                prompt=job_input['prompt'],
                num_inference_steps=job_input['refiner_inference_steps'],
                strength=job_input['strength'],
                image=init_image,
                generator=generator
            ).images
        else:
            logger.info("Starting base image generation")
            image = MODELS.base(
                prompt=job_input['prompt'],
                negative_prompt=job_input['negative_prompt'],
                height=job_input['height'],
                width=job_input['width'],
                num_inference_steps=job_input['num_inference_steps'],
                guidance_scale=job_input['guidance_scale'],
                denoising_end=job_input['high_noise_frac'],
                output_type="latent",
                num_images_per_prompt=job_input['num_images'],
                generator=generator
            ).images

            logger.info("Running refiner")
            try:
                output = MODELS.refiner(
                    prompt=job_input['prompt'],
                    num_inference_steps=job_input['refiner_inference_steps'],
                    strength=job_input['strength'],
                    image=image,
                    num_images_per_prompt=job_input['num_images'],
                    generator=generator
                ).images
            except RuntimeError as err:
                error_msg = f"RuntimeError in refiner: {err}\nTraceback: {traceback.format_exc()}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "refresh_worker": True
                }

        logger.info("Saving and uploading images")
        image_urls = _save_and_upload_images(output, job['id'])

        results = {
            "images": image_urls,
            "image_url": image_urls[0],
            "seed": job_input['seed']
        }

        if starting_image:
            results['refresh_worker'] = True

        logger.info("Job completed successfully")
        return results

    except Exception as e:
        error_msg = f"Error in generate_image: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return {"error": error_msg}

if __name__ == "__main__":
    logger.info("Starting RunPod worker")
    runpod.serverless.start({"handler": generate_image})
