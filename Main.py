import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import random


## TO DO ##
### GUI for changing prompt and other options
### Image to Image
### allow for non random seed
## fix need to add path for ckpt model file 
## add upscaling


DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")



base_path = Path(__file__).resolve().parent
# Construct paths relative to the script
vocab_file_path = base_path / "tokenizer_vocab.json"
merges_file_path = base_path / "tokenizer_merges.txt"
# Use the constructed paths in the tokenizer initialization
tokenizer = CLIPTokenizer(vocab_file_path, merges_file=merges_file_path)



model_file = "" #path to checkpoint model MUST BE.ckpt file



models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)



y = int(input("\nEnter Image Count "))

## TEXT TO IMAGE


prompt = input("\nEnter Prompt: ")
uncond_prompt = input("\nEnter Neg Prompt: ")  # negative prompt
do_cfg = True
cfg_scale = int(input("\nEnter cfg scale number - min1: max:14 :"))  # min: 1, max: 14



## SAMPLER

sampler = "ddpm"
num_inference_steps = 50

for x in range(y):

    seed = random.randint(1, 10000000000)

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        #input_image=input_image,
        #strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    # Combine the input image and the output image into a single image.
    Image.fromarray(output_image).show()
