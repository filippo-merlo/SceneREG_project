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
from torchvision import transforms
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

def remove_object(image, object_mask):
    return simple_lama(image, object_mask.convert('L'))

def generate_prompt_cogvlm2(tokenizer, model, image, obj, scene_category):
        # Text-only template
    text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
    image = image
    # Input user query
    query = f"Human: Describe the {obj}. Focus only on its appearence."

    # Format query
    if image is None:
        query = text_only_template.format(query)

    # Prepare input for model
    if image is None:
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, template_version='chat')
    else:
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, images=[image], template_version='chat')

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(torch.float16)]] if image is not None else None,
    }

    # Generate response
    gen_kwargs = {"max_new_tokens": 255, "pad_token_id": 128002}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|end_of_text|>")[0]
        #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    return image

def preprocess_mask(mask):
    mask = mask.convert("L")
    mask = transforms.CenterCrop((mask.size[1] // 64 * 64, mask.size[0] // 64 * 64))(mask)
    return mask

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
    
def generate_sd3(pipe, image, target_box, new_object, scene_category, prompt_obj_descr):
    size, _ = image.size
    print('SIZE:', size)
    x, y, w, h = target_box  # Coordinates and dimensions of the white box

    image = image.convert("RGB")

    # Step 3: Create the mask with the size of the new square image
    mask = np.zeros((size, size), dtype=np.float32)

    # Adjusting the region to fit within the image size limits
    x_end = min(x + w, size)
    y_end = min(y + h, size)
    mask[int(y):int(y_end), int(x):int(x_end)] = 1

    # Convert the mask to a black and white .png format (in memory, not saving to disk)
    mask_png_format = (mask * 255).astype(np.uint8)

    # Convert to a PIL image
    mask_image = Image.fromarray(mask_png_format)
    #mask = mask_image.convert("L")

    image = preprocess_image(image)
    mask = preprocess_mask(mask_image)

    if new_object[0] in ['a', 'e', 'i', 'o', 'u']:
        art = 'An'
    else:
        art = 'A'

    prompt = f"{art} {new_object}, realistic, center of the image, accurate, high quality, correct perspective."
    prompt_2 = f"{art} {new_object}, realistic, center of the image, accurate, high quality, correct perspective."
    prompt_3 = f"{art} {new_object}. {prompt_obj_descr}"
    
    with torch.no_grad():
        generated_image = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            image=image,
            mask_image=mask,
            height=size,
            width=size,
            num_inference_steps=50,
            guidance_scale=10,
            strength=1,
            padding_mask_crop = 256,
            num_images_per_prompt = 6
        ).images

    return generated_image, mask_image

def generate_sd3_from_patch(pipe, image, mask, bbox_in_mask, new_object, scene_category, prompt_obj_descr):
    size, _ = image.size
    _, _, w, h =  bbox_in_mask
    image = image.convert("RGB")
    #mask_png_format = (mask * 255).astype(np.uint8)

    # Convert to a PIL image
    #mask_image = Image.fromarray(mask_png_format)
    #mask = mask_image.convert("L")

    image = preprocess_image(image)
    mask = preprocess_mask(mask)

    if new_object[0] in ['a', 'e', 'i', 'o', 'u']:
        art = 'An'
    else:
        art = 'A'

    prompt = f"{art} {new_object}, (single object), (complete object), realistic, center of the image, accurate, high quality, correct perspective."
    prompt_2 = f"{art} {new_object}, (single object), (complete object), realistic, center of the image, accurate, high quality, correct perspective."
    prompt_3 = f"{art} {new_object}. {prompt_obj_descr}"

    print(int(w), int(h))
    with torch.no_grad():
        generated_image = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            image=image,
            mask_image=mask,
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=4.0,
            strength=0.7,
            padding_mask_crop = 0,
            num_images_per_prompt = 1
        ).images

    return generated_image

def threshold_image(image, threshold=1):
    # Ensure the image is in RGB mode
    image = image.convert("RGB")
    
    # Convert the image to grayscale
    grayscale_image = image.convert("L")
    
    # Apply the threshold to create a binary mask
    binary_mask = grayscale_image.point(lambda p: 255 if p > threshold else 0)
    
    # Create a new image for the result
    result_image = Image.new("RGB", image.size)
    
    # Iterate through pixels and apply the thresholding logic
    for x in range(image.width):
        for y in range(image.height):
            if binary_mask.getpixel((x, y)) == 255:
                result_image.putpixel((x, y), (255, 255, 255))  # White
            else:
                result_image.putpixel((x, y), (0, 0, 0))  # Black
    
    return result_image

def generate_silhouette_mask(pipe, image, target_box, new_object):
    size, _ = image.size
    print('SIZE:', size)
    target_box = adjust_rectangle(target_box, 0.5, 2)
    x, y, w, h = target_box  # Coordinates and dimensions of the white box

    # Step 3: Create the mask with the size of the new square image
    mask = np.zeros((size, size), dtype=np.float32)
    image_black_png = Image.fromarray(mask)

    # Adjusting the region to fit within the image size limits
    x_end = min(x + w, size)
    y_end = min(y + h, size)
    mask[int(y):int(y_end), int(x):int(x_end)] = 1

    # Convert the mask to a black and white .png format (in memory, not saving to disk)
    mask_png_format = (mask * 255).astype(np.uint8)

    # Convert to a PIL image
    mask_image = Image.fromarray(mask_png_format)

    image = preprocess_image(image_black_png)
    mask = preprocess_mask(mask_image)

    if new_object[0] in ['a', 'e', 'i', 'o', 'u']:
        art = 'an'
    else:
        art = 'a'

    prompt = f"White silhouette of {art} {new_object} on a black background. No black spaces in the silhouette."
    prompt_2 = f"White silhouette of {art} {new_object} on a black background. No black spaces in the silhouette."
    prompt_3 = f"White silhouette of {art} {new_object} on a black background. No black spaces in the silhouette."
    
    with torch.no_grad():
        generated_image = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            image=image,
            mask_image=mask,
            height=size,
            width=size,
            num_inference_steps=10,
            guidance_scale=10,
            strength=1,
            padding_mask_crop = 0,
            num_images_per_prompt = 1
        ).images

    generated_silohuette_mask = generated_image[0]
    silohuette_mask = threshold_image(generated_silohuette_mask, threshold=5)
    
    return silohuette_mask

def generate_sd3_from_silhouette(pipe, image, silohuette_mask, new_object, scene_category, prompt_obj_descr):
    size, _ = image.size
    image = preprocess_image(image)
    mask = preprocess_mask(silohuette_mask)

    if new_object[0] in ['a', 'e', 'i', 'o', 'u']:
        art = 'An'
    else:
        art = 'A'

    prompt = f"{art} {new_object}, realistic."
    prompt_2 = f"{art} {new_object}, realistic."
    prompt_3 = f"{art} {new_object}, realistic."
    
    with torch.no_grad():
        generated_image = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            image=image,
            mask_image=mask,
            height=size,
            width=size,
            num_inference_steps=50,
            guidance_scale=7,
            strength=0.8,
            padding_mask_crop = 20,
            num_images_per_prompt = 6
        ).images

    return generated_image


def generate_new_images(data, n):
    gen_images = n
    sets = []
    cogvlm2_tokenizer, cogvlm2_model = init_covlm2()
    for i in range(gen_images):
        try:
            # Get the masked image with target and scene category
            target, scene_category, image_picture, image_picture_w_bbox, target_bbox, cropped_target_only_image, object_mask = get_coco_image_data(data)
            # remove the object before background
            image_clean = remove_object(image_picture, object_mask)
            image_patch, image_patch_mask, patch_coord, bbox_in_mask = get_image_square_patch_rescaled(image_clean, target_bbox, 10)

            # SELECT OBJECT TO REPLACE
            objects_for_replacement_list = find_object_for_replacement(target, scene_category)
            images_names, images_paths = compare_imgs(cropped_target_only_image, objects_for_replacement_list)
            print(images_names)

            # Generate promt
            prompt_obj_descr = generate_prompt_cogvlm2(cogvlm2_tokenizer, cogvlm2_model, Image.open(images_paths[0]), images_names[0], scene_category)
            print(prompt_obj_descr)

            sets.append((image_patch, image_patch_mask, bbox_in_mask, target, scene_category, images_names, prompt_obj_descr))
        except Exception as e:
            print(e)

    import gc
    del cogvlm2_tokenizer, cogvlm2_model
    gc.collect()
    torch.cuda.empty_cache()

    pipe = init_sd3_model()
    
    for i, set in enumerate(sets):
        #try:
            image_patch, image_patch_mask, bbox_in_mask, target, scene_category, images_names, prompt_obj_descr = set
            
            # Inpainting the target
            generated_image = generate_sd3_from_patch(pipe, image_patch, image_patch_mask, bbox_in_mask, images_names[0], scene_category, prompt_obj_descr)
            # save the image
            
            save_path_target_mask = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_target_mask.jpg')
            image_patch_mask.save(save_path_target_mask)

            for i, image in enumerate(generated_image):
                save_path = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_replaced_{i}.jpg')
                image.save(save_path)
        #except Exception as e:
        #    print(e)




"""
            # SELECT OBJECT TO REPLACE
            objects_for_replacement_list = find_object_for_replacement(target, scene_category)
            images_names, images_paths = compare_imgs(cropped_target_only_image, objects_for_replacement_list)
            print(images_names)

            prompt_obj_descr = generate_prompt_cogvlm2(cogvlm2_tokenizer, cogvlm2_model, Image.open(images_paths[0]), images_names[0], scene_category)
            print(prompt_obj_descr)
            

            # ADD BACKGROUND
            #image_clean_with_background, image_mask_with_background, new_bbox, path_to_img = add_black_background(image_clean, object_mask, target_bbox)
            
            sets.append((image_clean_with_background, new_bbox, target, scene_category, images_names, prompt_obj_descr, image_mask_with_background))
        except Exception as e:
            print(e)

    import gc
    del cogvlm2_tokenizer, cogvlm2_model
    gc.collect()
    torch.cuda.empty_cache()

    pipe = init_sd3_model()

    for i, set in enumerate(sets):
        try:
            upscaled_image, upscaled_bbox, target, scene_category, images_names, prompt_obj_descr, image_mask_with_background = sets[i]
           
            # Inpainting the target
            generated_image, square_mask_image = generate_sd3(pipe, upscaled_image, upscaled_bbox, images_names[0], scene_category, prompt_obj_descr)
            # save the image
            
            save_path_target_mask = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_target_mask.jpg')
            image_mask_with_background.save(save_path_target_mask)

            save_path_original_clean = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_clean.jpg')
            upscaled_image.save(save_path_original_clean)

            save_path_square_mask = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_square_mask.jpg')
            square_mask_image.save(save_path_square_mask)

            for i, image in enumerate(generated_image):
                save_path = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_replaced_{i}.jpg')
                image.save(save_path)
        except Exception as e:
            print(e)

      
def try_things(data):

    # Get the masked image with target and scene category
    target, scene_category, image_picture, image_picture_w_bbox, target_bbox, cropped_target_only_image, image_mask = get_coco_image_data(data)
    # SELECT OBJECT TO REPLACE
    objects_for_replacement_list = find_object_for_replacement(target, scene_category)
    images_names, images_paths = compare_imgs(cropped_target_only_image, objects_for_replacement_list)
    print(images_names)
    # remove the object before background
    image_clean = remove_object(image_picture, image_mask.convert('L'))

    # ADD BACKGROUND
    image_clean_with_background, image_mask_with_background, new_bbox, path_to_img = add_black_background(image_clean, image_mask, target_bbox)

    pipe = init_sd3_model()

    # Inpainting the target
    image_silohuette_mask = generate_silhouette_mask(pipe, image_mask_with_background, new_bbox, images_names[0])
    # save the image
    
    save_path_target_mask = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_image_silohuette_mask.jpg')
    image_silohuette_mask.save(save_path_target_mask)

    save_path_square_mask = os.path.join(data_folder_path+'/generated_images', f'{scene_category.replace('/','_')}_{target.replace('/','_')}_{images_names[0].replace('/','_')}_square_mask.jpg')
    image_mask_with_background.save(save_path_square_mask)


"""