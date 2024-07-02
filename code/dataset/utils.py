#%%
import os 
from config import *
import torch
import math

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
    for key, value in dictionary.items():
        print(f"{ind}Key: [{key}], Type of Value: [{type(value).__name__}]")
        if isinstance(value, dict):
            ind2 = ind + '  '
            print_dict_structure(value, ind2)
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                ind2 = ind + '  '
                print_dict_structure(value[0], ind2)

def classify_scene_vit(image_picture, device, processor, model):
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

def subtract_in_bounds(x, y):
    if x - y > 0:
        return int(x - y) 
    else:
        return 0
    
def add_in_bounds(x, y, max):
    if x + y < max:
        return int(x + y)
    else:
        return int(max)

# FIND OBJECT TO REPLACE 

def find_object_to_replace(target_object_name, scene_name):
    scene_name = map_sun2ade[scene_name.replace('/', '_')]
    # get the more similar in size with the less semantic relatedness to the scene
    final_scores = []
    z_size_scores = []
    relatedness_scores = []
    for ade_name in map_ade2things.keys():
        # exclude objects that are labelled as typical for the scene by llama-3
        related = False
        for object in llama_norms[scene_name][ade_name]:
            if object[1]>object[2]:
                related = True
        if related:
            scene_relatedness_score = 100
        # ade object --> scene relatedness
        scene_relatedness_score = object_scene_rel_matrix.at[ade20k_object_names.index(ade_name), scene_name]
        #if scene_relatedness_score != 0:
        #    scene_relatedness_score = 100
        
        # not count objects appearing in only one scene 
        if ade_name in object_to_remove:
            scene_relatedness_score = 100
        

        #bert_score = bert_scene_object_rel_matrix.at[ade20k_object_names.index(ade_name), scene_name]
        ## target, coco object --> ade object --> emb
        #target_distr = object_scene_rel_matrix.iloc[ade20k_object_names.index(map_coco2ade[target_object_name][1])]
        ## non target, ade object --> emb
        #sde_name_distr = object_scene_rel_matrix.iloc[ade20k_object_names.index(ade_name)]
        ## target - non target cos sim
        #cos_sim = cosine_similarity(target_distr,sde_name_distr)

        # target size
        # coco obj --> ade obj --> things obj
        things_name_target = map_ade2things[map_coco2ade[target_object_name][1]]
        if target_object_name in things_name_target:
            target_idx = things_words.index(target_object_name)
            target_size_score = size_mean_matrix.at[target_idx, 'Size_mean']
            target_sd_size_score = size_mean_matrix.at[target_idx, 'Size_SD']
        else:
            target_idx = [things_words.index(n) for n in things_name_target]
            target_size_score = 0
            target_sd_size_score = 0
            for id in target_idx:
                target_size_score += size_mean_matrix.at[id, 'Size_mean']
                target_sd_size_score += size_mean_matrix.at[id, 'Size_SD']
            target_size_score = target_size_score/len(target_idx)
            target_sd_size_score = target_sd_size_score/len(target_idx)

        # ade obj size
        # ade obj --> things obj
        things_name_ade_name = map_ade2things[ade_name]
        if len(things_name_ade_name) == 1:
            ade_idx = things_words.index(things_name_ade_name[0])
            ade_size_score = size_mean_matrix.at[ade_idx, 'Size_mean']
            ade_sd_size_score = size_mean_matrix.at[ade_idx, 'Size_SD']
        else:
            ade_idx = [things_words.index(n) for n in things_name_ade_name]
            ade_size_score = 0
            ade_sd_size_score = 0
            for id in ade_idx:
                ade_size_score += size_mean_matrix.at[id, 'Size_mean']
                ade_sd_size_score += size_mean_matrix.at[id, 'Size_SD']
            ade_size_score = ade_size_score/len(ade_idx)
            ade_sd_size_score = ade_sd_size_score/len(ade_idx)

        z_size_score = abs((target_size_score - ade_size_score)/math.sqrt(target_sd_size_score**2 + ade_sd_size_score**2))

        total_score = z_size_score + scene_relatedness_score
        z_size_scores.append(z_size_score)
        relatedness_scores.append(scene_relatedness_score)
        if ade_name == map_coco2ade[target_object_name][1]:
            total_score = 100
        final_scores.append(total_score)

    # get top k lower scores idxs
    #kidxs_0, _ = lowest_k(final_scores, 100)
    #for i, _ in enumerate(final_scores):
    #    if i not in kidxs_0:
    #        final_scores[i] = 100
    #
    #final_scores = sum_lists(relatedness_scores, relatedness_scores) 
    kidxs, vals = lowest_k(final_scores, 20)
    print(vals)
    adeknames = [list(map_ade2things.keys())[i] for i in kidxs]
    print(adeknames)
    things_names = [map_ade2things[ade_name] for ade_name in adeknames]
    return things_names

import numpy as np
def sum_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    
    return [x + y for x, y in zip(list1, list2)]

def cosine_similarity(vec1, vec2):
    # Compute the dot product of the vectors
    dot_product = np.dot(vec1, vec2)
    
    # Compute the magnitudes of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cosine_sim

def lowest_k(alist, k):
    # Step 1: Enumerate the list to pair each element with its index
    enumerated_list = list(enumerate(alist))
    
    # Step 2: Sort the enumerated list by the element values
    sorted_list = sorted(enumerated_list, key=lambda x: x[1])
    
    # Step 3: Extract the indices of the first k elements
    lowest_k_indices = [index for index, value in sorted_list[:k]]
    lowest_k_values = [value for index, value in sorted_list[:k]]
    
    return lowest_k_indices, lowest_k_values

def highest_k(alist, k):
    # Step 1: Enumerate the list to pair each element with its index
    enumerated_list = list(enumerate(alist))
    
    # Step 2: Sort the enumerated list by the element values in descending order
    sorted_list = sorted(enumerated_list, key=lambda x: x[1], reverse=True)
    
    # Step 3: Extract the indices and values of the first k elements
    highest_k_indices = [index for index, value in sorted_list[:k]]
    highest_k_values = [value for index, value in sorted_list[:k]]
    
    return highest_k_indices, highest_k_values

# get object scene relatedness score
def object_scene_rel(object_name, scene_name):
    object_idx = map_coco2ade[object_name][0]
    relatedness_score = object_scene_rel_matrix.at[object_idx, scene_name].item()
    return relatedness_score

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


from transformers import AutoImageProcessor, ViTModel, ViTConfig 
import torch
vitc_image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vitc_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)


things_images_path = '/Users/filippomerlo/Desktop/Datasets/sceneREG_data/THINGS/THINGS/Images'

divisions = {
  "object_images_A-C": range(ord('a'), ord('c') + 1),  # a-c (inclusive)
  "object_images_D-K": range(ord('d'), ord('k') + 1),  # d-k (inclusive)
  "object_images_L-Q": range(ord('l'), ord('q') + 1),  # l-q (inclusive)
  "object_images_R-S": range(ord('r'), ord('s') + 1),  # r-s (inclusive)
  "object_images_T-Z": range(ord('t'), ord('z') + 1),  # t-z (inclusive)
}
# Convert ranges to lists of characters for clarity (optional)
letters = {}
for division, char_range in divisions.items():
  letters[division] = [chr(c) for c in char_range]
letter_to_division = reverse_dict(letters)

import re
from PIL import Image
def get_images_names(substitutes_list):
    # get things images paths [(name, path)...]
    things_names = list(set([n[0] for n in substitutes_list]))
    images_names_list = []
    images_path_list = []
    for object_name in things_names:
        folders_path = os.path.join(things_images_path, letter_to_division[object_name[0]])
        images_paths = get_all_names(folders_path)
        for i_p in images_paths:
            things_obj_name = re.sub(r"\d+", "",i_p.split('/')[-2]).replace('_',' ')
            if object_name == things_obj_name:
                images_names_list.append(object_name)
                images_path_list.append(i_p)
    return images_path_list, images_names_list


def compare_imgs(target_patch, substitutes_list):
    # get things images paths [(name, path)...]
    images_path_list, images_names_list = get_images_names(substitutes_list)

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
    # get top 5
    k = 5
    v, indices = torch.topk(torch.tensor(similarities), k)
    print([images_names_list[i] for i in indices])
    return [images_path_list[i] for i in indices]

from PIL import Image

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

import os

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

