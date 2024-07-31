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
