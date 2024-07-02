#%%
from config import *
from utils import *
from pprint import pprint
import json
from tqdm import tqdm
import random as rn
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter
from diffusers import AutoPipelineForInpainting

pipeline =  AutoPipelineForInpainting.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", cache_dir = CACHE_DIR_SHARED,  torch_dtype=torch.float16).to(device)
pipeline.enable_model_cpu_offload()

# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
generator = torch.Generator("cuda").manual_seed(92)

# get random image 
def get_random_image(data):
    while img_name == None:
        img_name = rn.choice(list(data.keys()))
        image = data[img_name]
        for fix in image['fixations']:
            if fix['condition'] == 'absent':
                target = None
                break
            if 'task' in fix.keys():
                target = fix['task']
                break
            else:
                target = None
        if target == None:
            img_name = None
            
    return image_picture


# check predictions
def generate(init_image, target_box, new_object):
    # Given data
    x, y, w, h = target_box  # Coordinates and dimensions of the white box
    max_w, max_h = init_image.size  # Size of the image
    # Create a black background image
    mask = Image.new("RGB", (max_w, max_h), "black")
    # Create a drawing object
    draw = ImageDraw.Draw(mask)
    # Define the coordinates of the white box
    left = x
    top = y
    right = x + w
    bottom = y + h
    # Draw the white box on the black background
    draw.rectangle([left, top, right, bottom], outline="white", fill="white")
    # Save or display the image
    blurred_mask = pipeline.mask_processor.blur(mask, blur_factor=33)
    prompt = f"a {new_object}, realistic, highly detailed, 8k"
    negative_prompt = "bad anatomy, deformed, ugly, disfigured"
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=blurred_mask, generator=generator).images[0]
    grid_image = make_image_grid([init_image, blurred_mask, image], rows=1, cols=3)
    grid_image.show()

def make_image_grid(images, rows, cols):
        """
        Create a grid of images.
        
        :param images: List of PIL Image objects.
        :param rows: Number of rows in the grid.
        :param cols: Number of columns in the grid.
        :return: PIL Image object representing the grid.
        """
        assert len(images) == rows * cols, "Number of images does not match rows * cols"

        # Get the width and height of the images
        width, height = images[0].size
        
        # Create a new blank image with the correct size
        grid_width = width * cols
        grid_height = height * rows
        grid_image = Image.new('RGB', (grid_width, grid_height))

        # Paste the images into the grid
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid_image.paste(img, (col * width, row * height))

        return grid_image
