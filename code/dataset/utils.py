### UTILS FILE WITH ALL THE BASIC FUNCTIONS USED

### IMPORTS
import os 
from config import *
import torch
import math
import numpy as np
from tqdm import tqdm
import random as rn
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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

def print_dict_structure(dictionary, ind = ''):
    """
    Visualize nested structire of a dictionary.
    """
    for key, value in dictionary.items():
        print(f"{ind}Key: [{key}], Type of Value: [{type(value).__name__}]")
        if isinstance(value, dict):
            ind2 = ind + '  '
            print_dict_structure(value, ind2)
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                ind2 = ind + '  '
                print_dict_structure(value[0], ind2)

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


def augment_area_within_bounds(coordinates, scale_factor, img_width, img_height):
    # Convert the list to a numpy array and reshape to a 2D array
    coords = np.array(coordinates).reshape(-1, 2)
    
    # Calculate the centroid
    centroid = np.mean(coords, axis=0)
    
    # Translate coordinates to the origin (centroid)
    translated_coords = coords - centroid
    
    # Initial scaling
    scaled_coords = translated_coords * scale_factor
    
    # Translate the coordinates back to the original centroid position
    augmented_coords = scaled_coords + centroid
    
    # Check if any point is out of bounds
    min_x, min_y = np.min(augmented_coords, axis=0)
    max_x, max_y = np.max(augmented_coords, axis=0)
    
    # Adjust scale factor if necessary
    if min_x < 0 or min_y < 0 or max_x > img_width or max_y > img_height:
        # Calculate the necessary scaling factors for both x and y directions
        scale_x = min(img_width / (max_x - centroid[0]), centroid[0] / -min_x) if min_x < 0 or max_x > img_width else scale_factor
        scale_y = min(img_height / (max_y - centroid[1]), centroid[1] / -min_y) if min_y < 0 or max_y > img_height else scale_factor
        
        # Use the smaller of the two scaling factors to ensure no out-of-bounds
        adjusted_scale_factor = min(scale_x, scale_y)
        
        # Re-scale with the adjusted scale factor
        scaled_coords = translated_coords * adjusted_scale_factor
        augmented_coords = scaled_coords + centroid
    
    return augmented_coords

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

        # Image processing and cropping code
        # Segment the target area in the image
        # Convert the image from RGB to BGR format using OpenCV
        image_mask_cv2 = cv2.cvtColor(np.array(image_picture), cv2.COLOR_RGB2BGR)

        # Get augmented segmentation coordinates
        max_w, max_h = image_picture.size
        target_segmentation = augment_area_within_bounds(target_segmentation, 1.15, max_w, max_h)

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
        max_w, max_h = image_picture.size

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
        #cropped_image = image_picture.crop((x_c, y_c, w_c, h_c))
       
        # Classify scene
        scene_category = classify_scene_vit(image_picture)
        return target, scene_category, image_picture, target_bbox, cropped_target_only_image_pil, image_mask_pil

### SCENE CLASSIFICATION
def classify_scene_vit(image_picture):
    """
    Classify an image with the classes of SUN397 using a Vision Transformer model.
    """
    inputs = vit_processor(image_picture, return_tensors="pt").to(device)
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
        target_idx = things_words_context.index(things_name_target)
        target_size_score = things_plus_size_mean_matrix.at[target_idx, 'Size_mean']
        target_sd_size_score = things_plus_size_mean_matrix.at[target_idx, 'Size_SD']
        
        # object size
        object_idx = things_words_context.index(thing)
        object_size_score = things_plus_size_mean_matrix.at[object_idx, 'Size_mean']
        object_sd_size_score = things_plus_size_mean_matrix.at[object_idx, 'Size_SD']
       
        size_distance = abs((target_size_score - object_size_score)/math.sqrt(target_sd_size_score**2 + object_sd_size_score**2))

        total_score = size_distance + scene_relatedness_score

        if thing == things_name_target or related:
            total_score = 100

        final_scores.append(total_score)

    kidxs, vals = select_k(final_scores, 12, lower = True)
    things_names = [things_words_context[i] for i in kidxs]
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
            image_input = vitc_image_processor(image, return_tensors="pt").to(device)
            image_outputs = vitc_model(**image_input)
            image_embeds = image_outputs.last_hidden_state[0][0].to('cpu')#.squeeze().mean(dim=1).to('cpu')
            images_embeddings.append(image_embeds)
        # embed target 
        target_input = vitc_image_processor(target_patch, return_tensors="pt").to(device)
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


def visualize_images(image_paths):
  """
  This function takes a list of image paths and displays them using PIL.

  Args:
      image_paths: A list containing paths to the images.
  """
  for path in image_paths:
    try:
      # Open the image using PIL
      image = Image.open(path)

      # Display the image using Image.show()
      image.show()
    except FileNotFoundError:
      print(f"Error: File not found: {path}")



def visualize_coco_image(self, img_name = None):
        
        if img_name != None:
            image = self.data[img_name]
        else:
            while img_name == None:
                img_name = rn.choice(list(self.data.keys()))
                image = self.data[img_name]
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
        images_paths = get_files(coco_images_path)
        image_picture = None
        for image_path in images_paths:
            if img_name in image_path:
                image_picture = Image.open(image_path)
                break
        # Convert PIL image to OpenCV format
        image_cv2 = cv2.cvtColor(np.array(image_picture), cv2.COLOR_RGB2BGR)

        # Draw the box of the image
        ann_key = 'instances_train2017_annotations'
        try:
            image[ann_key]
        except:
            ann_key = 'instances_val2017_annotations'

        target_bbox = None
        for ann in image[ann_key]:
            id = ann['category_id']
            color = (255, 0, 0)  # Red color
            object_names = list()
            for cat in coco_object_cat:
                if cat['id'] == id:
                    cat_name = cat['name']
                    object_names.append(cat_name)
            if target in object_names:
                color = (0, 0, 255)
                target_bbox = ann['bbox']
                target_segmentation = ann['segmentation']
                target_area = ann['area']
            x, y, width, height = ann['bbox']
            thickness = 2
            cv2.rectangle(image_cv2, (int(x), int(y)), (int(x + width), int(y + height)), color, thickness)

        # retrieve captions
        image_captions = []
        cap_key = 'captions_train2017_annotations'
        try:
            image[cap_key]
        except:
            cap_key = 'captions_val2017_annotations'
        for ann in image[cap_key]:
            caption = ann['caption']
            print(caption)
            image_captions.append(caption)

        # observe results
        # Convert back to PIL format for displaying
        image_with_box = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    
        # Display the image with the box
        plt.imshow(image_with_box)
        plt.axis('off')  # Turn off axis
        plt.show()

        # Crop
        # Segmentation
        image_mask_cv2 = cv2.cvtColor(np.array(image_picture), cv2.COLOR_RGB2BGR)
        target_segmentation = np.array(target_segmentation, dtype=np.int32).reshape((-1, 2))
        # Create a mask
        target_mask = np.zeros(image_mask_cv2.shape[:2], dtype=np.uint8)
        cv2.fillPoly(target_mask, [target_segmentation], 255)
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image_mask_cv2, image_mask_cv2, mask=target_mask)
        # Crop image 
        # Box
        x,y,w,h = target_bbox
        max_w, max_h = image_picture.size
        x_c = subtract_in_bounds(x,20)
        y_c = subtract_in_bounds(y,20)
        w_c = add_in_bounds(x,w+20,max_w)
        h_c = add_in_bounds(y,h+20,max_h)
        cropped_masked_image = masked_image[y_c:h_c, x_c:w_c]
        # Step 3: Convert the cropped image from BGR to RGB
        cropped_masked_image_rgb = cv2.cvtColor(cropped_masked_image, cv2.COLOR_BGR2RGB)
        # Step 4: Convert the cropped image to a PIL image
        cropped_masked_image_pil = Image.fromarray(cropped_masked_image_rgb)
        # Show
        plt.imshow(cropped_masked_image_pil)
        plt.axis('off')  # Turn off axis
        plt.show()

        cropped_image = image_picture.crop((x_c,y_c,w_c,h_c))
       
        # Classify scene
        #classify_scene_clip_llava(image_picture, scene_labels_context)
        scene_category = classify_scene_vit(image_picture)
        print(scene_category)
        # retrieve info from obscene
        objects_to_replace = find_object_to_replace(target, scene_category)
        print(objects_to_replace)
        images_paths = compare_imgs(cropped_masked_image_pil, objects_to_replace)
        #generate(image_picture, target_bbox, objects_to_replace[0])
        visualize_images(images_paths)

def get_scene_predictions(self):
    all_predictions = []
    all_img_paths = []
    c = 0
    
    for img_name in tqdm(list(self.data.keys())):
        try:
            image = self.data[img_name]
            for fix in image['fixations']:
                if fix['condition'] == 'absent':
                    target = None
                    break
                if 'task' in fix.keys():
                    target = fix['task']
                    break
                else:
                    target = None
                
            images_paths = get_files(images_path)
            image_picture = None

            for image_path in images_paths:
                if img_name in image_path:
                    image_picture = Image.open(image_path)
                    all_img_paths.append(image_path)
                    break
            label = classify_scene_vit(image_picture)
            all_predictions.append(label)
        except:
            c += 1
            continue
    count = Counter(all_predictions)
    print(c)
    label_with_paths = dict()
    for i, lab in enumerate(all_predictions):
        if lab not in label_with_paths.keys():
            label_with_paths[lab] = list()
        label_with_paths[lab].append(all_img_paths[i])
    return count, label_with_paths

### GEenerate image function

def preprocess_mask(mask):
        mask = mask.convert("L")
        mask = transforms.CenterCrop((mask.size[1] // 64 * 64, mask.size[0] // 64 * 64))(mask)
        mask = transforms.ToTensor()(mask)
        mask = mask.to(device_gen)
        return mask

def preprocess_image(image):
        image = image.convert("RGB")
        image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0).to(device_gen)
        return image

import io
import base64

def encode_image_for_api(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    byte_data = buffer.getvalue()
    # Encode the byte stream to a Base64 string
    encoded_image = base64.b64encode(byte_data).decode('utf-8')
    return encoded_image

from gradio_client import Client

def api_upscale_image_gradio(image, path_to_image, scale_up_factor=2):
    client = Client("https://bookbot-image-upscaling-playground.hf.space/")
    encoded_image = encode_image_for_api(image)
    result = client.predict(
            path_to_image,	
            f"modelx{scale_up_factor}",	# str in 'Choose Upscaler' Radio component
            api_name="/predict"
    )
    new_image = Image.open(result)
    return new_image

def add_black_background(image, image_mask, target_box):
    x, y, w, h = target_box  # Coordinates and dimensions of the white box
    max_w, max_h = image.size 

    # Step 1: Add a black background to make the image square
    new_size = max(max_w, max_h)
    new_image = Image.new("RGB", (new_size, new_size), (0, 0, 0))
    new_image_mask = Image.new("RGB", (new_size, new_size), (0, 0, 0))
    offset_x = (new_size - max_w) // 2
    offset_y = (new_size - max_h) // 2
    new_image.paste(image, (offset_x, offset_y))
    new_image_mask.paste(image_mask, (offset_x, offset_y))

    # Step 2: Adjust the coordinates of the bounding box
    new_x = x + offset_x
    new_y = y + offset_y

    # The adjusted bounding box
    adjusted_box = (new_x, new_y, w, h)

    # save temporarely image:
    path = os.path.join(data_folder_path, 'temp.jpg')
    new_image.save(path)

    return new_image, new_image_mask, adjusted_box, path

def remove_object(image, masked_image):
    return simple_lama(image, masked_image)

def generate_sd3(image, target_box, new_object):
    # the image is square so ill get only one dimension
    size, _ = image.size
    print('SIZE:', size)
    x, y, w, h = target_box  # Coordinates and dimensions of the white box

    source = preprocess_image(image)

    # Step 3: Create the mask with the size of the new square image
    mask = np.zeros((size, size), dtype=np.float32)

    # Adjusting the region to fit within the image size limits
    x_end = min(x + w, size)
    y_end = min(y + h, size)
    mask[int(y):int(y_end), int(x):int(x_end)] = 1

    # Convert the mask to a black and white .png format (in memory, not saving to disk)
    mask_png_format = (mask * 255).astype(np.uint8)

    # Optional: Convert to a PIL image to visualize
    mask_image = Image.fromarray(mask_png_format)

    mask = preprocess_mask(
        mask_image
    )

    prompt = f"a high quality photography of a {new_object} taken with Canon EOS R3a"
    negative_prompt = f"worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"

    generated_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=source,
        mask_image=mask,
        height=size,
        width=size,
        num_inference_steps=100,
        strength=1,
        guidance_scale=1,
    ).images

    return generated_image, mask_image

# GET SUBSTITUTE
def generate_new_image(data):
    try:
        # Get the masked image with target and scene category
        target, scene_category, image_picture, target_bbox, cropped_target_only_image, image_mask = get_coco_image_data(data)
        
        # SELECT OBJECT TO REPLACE
        objects_for_replacement_list = find_object_for_replacement(target, scene_category)
        images_names, images_paths = compare_imgs(cropped_target_only_image, objects_for_replacement_list)
        print(images_names)

        # ADD BACKGROUND
        image_with_background, image_mask_with_background, new_bbox, path = add_black_background(image_picture, image_mask, target_bbox)

        image_with_background_clean = remove_object(image_with_background, image_mask_with_background.convert('L'))
        
        save_path_original_mask = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_mask.jpg')
        image_mask_with_background.save(save_path_original_mask)
        save_path_original_clean = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_clean.jpg')
        image_with_background_clean.save(save_path_original_clean)
        # upscale image and update bbox
        #scale_up_factor = 2
        #upscaled_image = api_upscale_image_gradio(image_with_background_clean, path, scale_up_factor)
        #upscaled_bbox = [x*scale_up_factor for x in new_bbox]
    except:
        print('Error')
        '''
        # and mask
        # Get the current size of the image
        size, _ = image_mask_with_background.size
        # Define the new size (scale by 2)
        new_size = (size * 2, size * 2)
        # Resize the image
        upscaled_image_mask_with_background = image_mask_with_background.resize(new_size, Image.Resampling.LANCZOS)
        temp_path = os.path.join(data_folder_path, 'temp_.png')
        upscaled_image_mask_with_background.save(temp_path)
        upscaled_image_mask_with_background = Image.open(temp_path).convert('L')

        save_path_original = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_original.jpg')
        upscaled_image_with_background.save(save_path_original)
        
        # Remove the object with LaMa
        clean_upscaled_image = remove_object(upscaled_image_with_background, upscaled_image_mask_with_background)
        

        save_path_original_clean = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_clean.jpg')
    
        clean_upscaled_image.save(save_path_original_clean)

        print(upscaled_image_with_background)
        print(clean_upscaled_image)

        # Inpainting the target
        generated_image, square_mask_image = generate_sd3(upscaled_image, upscaled_bbox, images_names[0])
        # save the image
        #save_path_original = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_original.jpg')
        #upscaled_image_with_background.save(save_path_original)

        #save_path_original_clean = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_clean.jpg')
        #clean_upscaled_image.save(save_path_original_clean)

        #save_path_round_mask = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_round_mask.jpg')
        #upscaled_image_mask_with_background.save(save_path_round_mask)

        save_path_square_mask = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_square_mask.jpg')
        square_mask_image.save(save_path_square_mask)

        for i, image in enumerate(generated_image):
            save_path = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_{i}.jpg')
            image.save(save_path)
    except:
       print('Error')

'''