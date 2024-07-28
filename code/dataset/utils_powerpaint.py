### UTILS FILE WITH ALL THE BASIC FUNCTIONS USED

### IMPORTS
import os 
from config_powerpaint import *
import torch
import math
import numpy as np
from tqdm import tqdm
import random as rn
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from collections import Counter
import re
from PIL import Image

### GENERAL FUNCTIONS

def get_files(directory):
    """
    Get all files in a directory with specified extensions.

    Args:
    - directory (str): The directory path.
    - extensions (list): A list of extensions to filter files by.

    Returns:
    - files (list): A list of file paths.
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(tuple([".json",".jpg"])):
            files.append(os.path.join(directory, file))
    return files

def reverse_dict(data):
    """
    This function reverses a dictionary by swapping keys and values.

    Args:
        data: A dictionary to be reversed.

    Returns:
        A new dictionary where keys become values and vice versa, handling duplicates appropriately.
    """
    reversed_dict = {}
    for key, value in data.items():
        for l in value:
            reversed_dict[str(l)] = key
    return reversed_dict

def subtract_in_bounds(x, y):
    """
    Subtract two numbers and ensure the result is non-negative.
    """
    if x - y > 0:
        return int(x - y) 
    else:
        return 0
    
def add_in_bounds(x, y, max):
    """
    Add two numbers and ensure the result is within a specified range.
    """
    if x + y < max:
        return int(x + y)
    else:
        return int(max)

def sum_lists(list1, list2):
    """
    Sum two lists element-wise.
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    
    return [x + y for x, y in zip(list1, list2)]

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    """
    # Compute the dot product of the vectors
    dot_product = np.dot(vec1, vec2)
    
    # Compute the magnitudes of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cosine_sim

def select_k(alist, k, lower = True):
    """
    Find the indices and values of the k lowest/higest elements in a list.
    """
    # Step 1: Enumerate the list to pair each element with its index
    enumerated_list = list(enumerate(alist))
    
    # Step 2: Sort the enumerated list by the element values
    if lower:
        reverse = False
    else:
        reverse = True
    sorted_list = sorted(enumerated_list, key=lambda x: x[1], reverse=reverse)
    
    # Step 3: Extract the indices of the first k elements
    k_indices = [index for index, value in sorted_list[:k]]
    k_values = [value for index, value in sorted_list[:k]]
    
    return k_indices, k_values


import numpy as np
from scipy.spatial import ConvexHull

def augment_area_within_bounds(coordinates, scale_factor, img_width, img_height):
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    
    # Flatten the array if it's a list of coordinate pairs
    if coordinates.ndim == 2 and coordinates.shape[1] == 2:
        coords = coordinates
    else:
        coords = coordinates.reshape(-1, 2)
    
    # Calculate the convex hull
    hull = ConvexHull(coords)
    hull_coords = coords[hull.vertices]
    
    # Calculate the centroid
    centroid = np.mean(hull_coords, axis=0)
    
    # Translate coordinates to the origin (centroid)
    translated_coords = hull_coords - centroid
    
    # Convert to polar coordinates
    radii = np.linalg.norm(translated_coords, axis=1)
    angles = np.arctan2(translated_coords[:, 1], translated_coords[:, 0])
    
    # Scale the radial distances
    scaled_radii = radii * scale_factor
    
    # Convert back to Cartesian coordinates
    scaled_coords = np.column_stack((scaled_radii * np.cos(angles), scaled_radii * np.sin(angles)))
    
    # Translate the coordinates back to the original centroid position
    augmented_coords = scaled_coords + centroid
    
    # Check if any point is out of bounds
    min_x, min_y = np.min(augmented_coords, axis=0)
    max_x, max_y = np.max(augmented_coords, axis=0)
    
    # Adjust scale factor if necessary
    if min_x < 0 or min_y < 0 or max_x > img_width or max_y > img_height:
        scale_x = min(img_width / (max_x - centroid[0]), centroid[0] / -min_x) if min_x < 0 or max_x > img_width else scale_factor
        scale_y = min(img_height / (max_y - centroid[1]), centroid[1] / -min_y) if min_y < 0 or max_y > img_height else scale_factor
        adjusted_scale_factor = min(scale_x, scale_y)
        scaled_radii = radii * adjusted_scale_factor
        scaled_coords = np.column_stack((scaled_radii * np.cos(angles), scaled_radii * np.sin(angles)))
        augmented_coords = scaled_coords + centroid

    return augmented_coords.astype(np.int32)

### GET COCO IMAGE DATA
def get_coco_image_data(data, img_name = None):
        
        # Get a random image from data
        if img_name != None:
            image = data[img_name]
        else:
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
                
        print('*',target)

        # Get the image picture
        images_paths = get_files(coco_images_path)
        image_picture = None
        for image_path in images_paths:
            if img_name in image_path:
                image_picture = Image.open(image_path)
                break
            
        # Get the target info 
        # get the right annotation key
        try:
            ann_key = 'instances_train2017_annotations'
            image[ann_key]
        except:
            ann_key = 'instances_val2017_annotations'

        # go through annotations (objects) for the image
        # every ann is an object
        for ann in image[ann_key]:
            id = ann['category_id'] # get the category id
            
            # get the target object info
            object_name = ''
            for cat in coco_object_cat:
                if cat['id'] == id:
                    object_name = cat['name']
            # get target object info
            if object_name == target:
                target_bbox = ann['bbox']
                target_segmentation = ann['segmentation']
                target_area = ann['area']

        if target_bbox:
            image_picture_w_bbox = Image.open(image_path)
            draw = ImageDraw.Draw(image_picture_w_bbox)
            x, y, width, height = target_bbox
            draw.rectangle([x, y, x + width, y + height], outline="red", width=3)

        # Image processing and cropping code
        # Segment the target area in the image
        # Convert the image from RGB to BGR format using OpenCV
        image_mask_cv2 = cv2.cvtColor(np.array(image_picture), cv2.COLOR_RGB2BGR)

        # Get augmented segmentation coordinates
        max_w, max_h = image_picture.size
        target_segmentation = augment_area_within_bounds(target_segmentation, 1.20, max_w, max_h)

        # Create a mask with the same height and width as the image, initialized to zeros (black)
        image_mask = np.zeros(image_mask_cv2.shape[:2], dtype=np.uint8)

        # Fill the polygon defined by target_segmentation on the mask with white (255)
        cv2.fillPoly(image_mask, [target_segmentation], 255)

        # Convert the mask to a PIL image
        image_mask_pil = Image.fromarray(image_mask)

        # Apply the mask to the image, resulting in an image where only the segmented area is visible
        target_only_image = cv2.bitwise_and(image_mask_cv2, image_mask_cv2, mask=image_mask)

        # Crop the image to a bounding box around the segmented area
        # Extract the bounding box coordinates and dimensions
        x, y, w, h = target_bbox

        # Adjust the coordinates and dimensions to include some padding
        x_c = subtract_in_bounds(x, 20)
        y_c = subtract_in_bounds(y, 20)
        w_c = add_in_bounds(x, w + 20, max_w)
        h_c = add_in_bounds(y, h + 20, max_h)

        # Crop the masked image using the adjusted coordinates and dimensions
        cropped_target_only_image = target_only_image[y_c:h_c, x_c:w_c]

        # Convert the cropped image from BGR to RGB format
        cropped_target_only_image_rgb = cv2.cvtColor(cropped_target_only_image, cv2.COLOR_BGR2RGB)

        # Convert the cropped image to a PIL image
        cropped_target_only_image_pil = Image.fromarray(cropped_target_only_image_rgb)

        # Additionally, crop the original image (without the mask) to the same bounding box

        # Classify scene
        scene_category = classify_scene_vit(image_picture)
        return target, scene_category, image_picture, image_picture_w_bbox, target_bbox, cropped_target_only_image_pil, image_mask_pil

### SCENE CLASSIFICATION
def classify_scene_vit(image_picture):
    """
    Classify an image with the classes of SUN397 using a Vision Transformer model.
    """
    inputs = vit_processor(image_picture, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = vit_model(**inputs).logits

    # Get the top 5 predictions
    top5_prob, top5_indices = torch.topk(logits, 5)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(top5_prob, dim=-1)

    # Get the labels for the top 5 indices
    top5_labels = [vit_model.config.id2label[idx.item()] for idx in top5_indices[0]]

    # Print the top 5 labels and their corresponding probabilities
    #for label, prob in zip(top5_labels, probabilities[0]):
    #    print(f"{label}: {prob:.4f}")
    probabilities = probabilities[0].to('cpu').numpy()

    return top5_labels[0]

# FIND OBJECT TO REPLACE 

def find_object_for_replacement(target_object_name, scene_name):
    # get the more similar in size with the less semantic relatedness to the scene
    final_scores = []

    for thing in things_words_context:
        # exclude objects that are labelled as typical for the scene by llama-3
        related = False
        for object in llama_norms[scene_name][thing]:
            if object[1]>object[2]:
                related = True
        if related:
            scene_relatedness_score = 100
        else:
            scene_relatedness_score = 0

        # target size
        things_name_target = map_coco2things[target_object_name]
        target_size_score = things_plus_size_mean_matrix[things_plus_size_mean_matrix['WordContext']==things_name_target]['Size_mean'].values[0]
        
        # object size
        object_size_score = things_plus_size_mean_matrix[things_plus_size_mean_matrix['WordContext']==thing]['Size_mean'].values[0]
        
        # modify to get only smaller objects
        #size_distance = abs((target_size_score - object_size_score)/math.sqrt(target_sd_size_score**2 + object_sd_size_score**2))
        size_distance = (target_size_score - object_size_score)#/math.sqrt(target_sd_size_score**2 + object_sd_size_score**2)
        if size_distance < 0:
            size_distance = 100

        total_score = size_distance + scene_relatedness_score

        if thing == things_name_target or related:
            total_score = 100

        final_scores.append(total_score)

    kidxs, vals = select_k(final_scores, 10, lower = True)
    things_names = [things_words_context[i] for i in kidxs]
    print(things_names)
    print(vals)
    return things_names

def get_images_names(substitutes_list):
    # get things images paths [(name, path)...]
    things_folder_names = list(set([things_words_id[things_words_context.index(n)] for n in substitutes_list]))
    images_names_list = []
    images_path_list = []
    for folder_name in things_folder_names:
        folders_path = os.path.join(things_images_path, folder_name)
        images_paths = get_all_names(folders_path)
        for i_p in images_paths:
            things_obj_name = re.sub(r"\d+", "",i_p.split('/')[-2]).replace('_',' ')
            if folder_name == things_obj_name:
                images_names_list.append(things_words_context[things_words_id.index(folder_name)].replace('/','_'))
                images_path_list.append(i_p)
    return images_path_list, images_names_list

def get_all_names(path):
    """
    This function retrieves all file and folder names within a directory and its subdirectories.

    Args:
        path: The directory path to search.

    Returns:
        A list containing all file and folder names.
    """
    names = []
    for root, dirs, files in os.walk(path):
        for name in files:
            names.append(os.path.join(root, name))
        for name in dirs:
            names.append(os.path.join(root, name))
    return names

def compare_imgs(target_patch, substitutes_list):
    # get things images paths [(name, path)...]
    images_path_list, images_names_list = get_images_names(substitutes_list)
    print(len(images_path_list))
    # embed images
    images_embeddings = []
    with torch.no_grad():
        for i_path in images_path_list:
            image = Image.open(i_path)
            image_input = vitc_image_processor(image, return_tensors="pt").to(DEVICE)
            image_outputs = vitc_model(**image_input)
            image_embeds = image_outputs.last_hidden_state[0][0].to('cpu')#.squeeze().mean(dim=1).to('cpu')
            images_embeddings.append(image_embeds)
        # embed target 
        target_input = vitc_image_processor(target_patch, return_tensors="pt").to(DEVICE)
        target_outputs = vitc_model(**target_input)
        target_embeds = target_outputs.last_hidden_state[0][0].to('cpu')#.squeeze().mean(dim=1).to('cpu')

    # compare
    similarities = []
    for i_embed in images_embeddings:
        similarities.append(cosine_similarity(target_embeds.detach().numpy(), i_embed.detach().numpy()))
    # get top k
    k = 5
    print(torch.tensor(similarities).size())
    v, indices = torch.topk(torch.tensor(similarities), k)
   
    return [images_names_list[i] for i in indices], [images_path_list[i] for i in indices]

### GEenerate image function
from gradio_client import Client

def api_upscale_image_gradio(image, scale_factor=4):
    # save temporarely image:
    path = os.path.join(data_folder_path, 'temp.jpg')
    image.save(path)
    # upscale image
    client = Client("https://bookbot-image-upscaling-playground.hf.space/")
    
    result = client.predict(
            path,	
            f"modelx{scale_factor}",	# str in 'Choose Upscaler' Radio component
            api_name="/predict"
    )
    new_image = Image.open(result)
    return new_image

def remove_object(image, object_mask):
    return simple_lama(image, object_mask.convert('L'))

def adjust_ratio(image, bbox, min_ratio, max_ratio):
    # Image size
    width, height = image.size
    
    # Calculate current aspect ratio
    x, y, w, h = bbox
    current_ratio = w / h
    
    # Adjust dimensions to fit within the desired aspect ratio range
    if current_ratio < min_ratio:
        # Adjust width to meet min_ratio
        new_w = min_ratio * h
        new_h = h
    elif current_ratio > max_ratio:
        # Adjust height to meet max_ratio
        new_w = w
        new_h = w / max_ratio
    else:
        # No adjustment needed
        return (x, y, w, h)
    
    # Center the new bounding box
    new_x = x + (w - new_w) / 2
    new_y = y + (h - new_h) / 2
    
    # Ensure the bounding box stays within the image boundaries
    if new_x < 0:
        new_x = 0
    elif new_x + new_w > width:
        new_x = width - new_w
    
    if new_y < 0:
        new_y = 0
    elif new_y + new_h > height:
        new_y = height - new_h
    
    return (int(new_x), int(new_y), int(new_w), int(new_h))

def get_image_square_patch_rescaled(image, target_bbox, padding):
    # Assuming `image`, `target_bbox`, and `adjust_ratio` are defined elsewhere in your code
    width, height = image.size
    new_x, new_y, new_w, new_h = adjust_ratio(image, target_bbox, 0.5, 2)
    
    # Ensure the bounding box dimensions are at least min_size
    side_length = max(new_w + padding * 2, new_h + padding * 2)

    # Adjust the top-left corner of the bounding box to fit within the image
    square_x = max(0, new_x + new_w // 2 - side_length // 2)
    square_y = max(0, new_y + new_h // 2 - side_length // 2)

    # Ensure the square does not go out of the right edge
    if square_x + side_length > width:
        square_x = width - side_length

    # Ensure the square does not go out of the bottom edge
    if square_y + side_length > height:
        square_y = height - side_length

    # If the side length is larger than the image dimensions, adjust it
    side_length = min(side_length, width, height)

    # Adjust side_length to be the nearest number divisible by 64
    numbers = [64, 128, 256, 512]
    side_length = min(numbers, key=lambda x: abs(x - side_length))
    #side_length =  (side_length + 64 - 1) // 64 * 64

    # Ensure the square does not go out of the right edge after adjustment
    if square_x + side_length > width:
        square_x = width - side_length

    # Ensure the square does not go out of the bottom edge after adjustment
    if square_y + side_length > height:
        square_y = height - side_length

    # Define the patch
    patch_coords = (square_x, square_y, square_x + side_length, square_y + side_length)

    # Ensure patch coordinates are valid
    patch_coords = (max(0, patch_coords[0]), max(0, patch_coords[1]), min(width, patch_coords[2]), min(height, patch_coords[3]))

    # Crop the image
    image_patch = image.crop(patch_coords)

    # upscale patch
    patch_size, _ = image_patch.size
    n_upscale = 1024/patch_size
    if n_upscale == 8:
        image_patch = api_upscale_image_gradio(image_patch, scale_factor=2)
        image_patch = api_upscale_image_gradio(image_patch, scale_factor=4)
    elif n_upscale == 4:
        image_patch = api_upscale_image_gradio(image_patch, scale_factor=4)
    elif n_upscale == 2:
        image_patch = api_upscale_image_gradio(image_patch, scale_factor=2)

    # Create the mask
    mask = Image.new('L', (1024, 1024), 0)
    draw = ImageDraw.Draw(mask)
    bbox_in_mask = (
        max(0, new_x - square_x)*n_upscale,
        max(0, new_y - square_y)*n_upscale,
        min(side_length, new_x - square_x + new_w)*n_upscale,
        min(side_length, new_y - square_y + new_h)*n_upscale
    )
    draw.rectangle(bbox_in_mask, outline=255, fill=255)

    return image_patch, mask, patch_coords, bbox_in_mask
    
def generate_new_images(data, n):
    for i in range(n):
        # Get the masked image with target and scene category
        target, scene_category, image_picture, image_picture_w_bbox, target_bbox, cropped_target_only_image, object_mask = get_coco_image_data(data)
        # remove the object before background
        image_clean = remove_object(image_picture, object_mask)
        image_patch, image_patch_mask, patch_coord, bbox_in_mask = get_image_square_patch_rescaled(image_clean, target_bbox, 100)
        image_patch_mask = image_patch_mask.convert('RGB')

        input_image = {
            "image": image_patch,
            "mask": image_patch_mask
        }
        # SELECT OBJECT TO REPLACE
        objects_for_replacement_list = find_object_for_replacement(target, scene_category)
        images_names, images_paths = compare_imgs(cropped_target_only_image, objects_for_replacement_list)
        print(images_names)
        if images_names[0][0] in ['a','e','i','o','u']:
            art = 'an'
        else:
            art = 'a'
        prompt = f"{art} {images_names[0]}"
        text_guided_prompt = prompt
        text_guided_negative_prompt = ''
        shape_guided_prompt = prompt
        shape_guided_negative_prompt = ''
        fitting_degree = 0.65 # 0-1
        ddim_steps = 45 # 1-50
        scale = 7.5 # 1-30
        seed = 666 # 0-2147483647
        task = "shape-guided"
        vertical_expansion_ratio = 2 #1-4
        horizontal_expansion_ratio = 2 #1-4
        outpaint_prompt = ''
        outpaint_negative_prompt = ''
        removal_prompt = ''
        removal_negative_prompt = ''

        # Inpainting the target
        dict_out, dict_res = controller.infer(
            input_image,
            text_guided_prompt,
            text_guided_negative_prompt,
            shape_guided_prompt,
            shape_guided_negative_prompt,
            fitting_degree,
            ddim_steps,
            scale,
            seed,
            task,
            vertical_expansion_ratio,
            horizontal_expansion_ratio,
            outpaint_prompt,
            outpaint_negative_prompt,
            removal_prompt,
            removal_negative_prompt,
        )

        save_path = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_original.jpg')
        image_picture.save(save_path)
        
        save_path = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_generated.jpg')
        dict_out[0].save(save_path)

            


