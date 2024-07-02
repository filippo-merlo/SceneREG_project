#%%
import os 
from config import *
import torch
import math

# Is MPS even available? macOS 12.3+
device = torch.device("mps")

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


# vit model 
import wandb
from transformers import ViTForImageClassification, AutoImageProcessor

# Define the scene labels
scene_labels_vit = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'apartment_building/outdoor', 'apse/indoor', 'aquarium', 'aqueduct', 'arch', 'archive', 'arrival_gate/outdoor', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'athletic_field/outdoor', 'atrium/public', 'attic', 'auditorium', 'auto_factory', 'badlands', 'badminton_court/indoor', 'baggage_claim', 'bakery/shop', 'balcony/exterior', 'balcony/interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'basketball_court/outdoor', 'bathroom', 'batters_box', 'bayou', 'bazaar/indoor', 'bazaar/outdoor', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'bistro/indoor', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth/indoor', 'botanical_garden', 'bow_window/indoor', 'bow_window/outdoor', 'bowling_alley', 'boxing_ring', 'brewery/indoor', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'cabin/outdoor', 'cafeteria', 'campsite', 'campus', 'canal/natural', 'canal/urban', 'candy_store', 'canyon', 'car_interior/backseat', 'car_interior/frontseat', 'carrousel', 'casino/indoor', 'castle', 'catacomb', 'cathedral/indoor', 'cathedral/outdoor', 'cavern/indoor', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'chicken_coop/indoor', 'chicken_coop/outdoor', 'childs_room', 'church/indoor', 'church/outdoor', 'classroom', 'clean_room', 'cliff', 'cloister/indoor', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'control_tower/outdoor', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'covered_bridge/exterior', 'creek', 'crevasse', 'crosswalk', 'cubicle/office', 'dam', 'delicatessen', 'dentists_office', 'desert/sand', 'desert/vegetation', 'diner/indoor', 'diner/outdoor', 'dinette/home', 'dinette/vehicle', 'dining_car', 'dining_room', 'discotheque', 'dock', 'doorway/outdoor', 'dorm_room', 'driveway', 'driving_range/outdoor', 'drugstore', 'electrical_substation', 'elevator/door', 'elevator/interior', 'elevator_shaft', 'engine_room', 'escalator/indoor', 'excavation', 'factory/indoor', 'fairway', 'fastfood_restaurant', 'field/cultivated', 'field/wild', 'fire_escape', 'fire_station', 'firing_range/indoor', 'fishpond', 'florist_shop/indoor', 'food_court', 'forest/broadleaf', 'forest/needleleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'garage/indoor', 'garbage_dump', 'gas_station', 'gazebo/exterior', 'general_store/indoor', 'general_store/outdoor', 'gift_shop', 'golf_course', 'greenhouse/indoor', 'greenhouse/outdoor', 'gymnasium/indoor', 'hangar/indoor', 'hangar/outdoor', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'hot_tub/outdoor', 'hotel/outdoor', 'hotel_room', 'house', 'hunting_lodge/outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'ice_skating_rink/indoor', 'ice_skating_rink/outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn/outdoor', 'islet', 'jacuzzi/indoor', 'jail/indoor', 'jail_cell', 'jewelry_shop', 'kasbah', 'kennel/indoor', 'kennel/outdoor', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'labyrinth/outdoor', 'lake/natural', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'library/indoor', 'library/outdoor', 'lido_deck/outdoor', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'market/indoor', 'market/outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'moat/water', 'monastery/outdoor', 'mosque/indoor', 'mosque/outdoor', 'motel', 'mountain', 'mountain_snowy', 'movie_theater/indoor', 'museum/indoor', 'music_store', 'music_studio', 'nuclear_power_plant/outdoor', 'nursery', 'oast_house', 'observatory/outdoor', 'ocean', 'office', 'office_building', 'oil_refinery/outdoor', 'oilrig', 'operating_room', 'orchard', 'outhouse/outdoor', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage/indoor', 'parking_garage/outdoor', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pilothouse/indoor', 'planetarium/outdoor', 'playground', 'playroom', 'plaza', 'podium/indoor', 'podium/outdoor', 'pond', 'poolroom/establishment', 'poolroom/home', 'power_plant/outdoor', 'promenade_deck', 'pub/indoor', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall/indoor', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'stadium/baseball', 'stadium/football', 'stage/indoor', 'staircase', 'street', 'subway_interior', 'subway_station/platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_pool/indoor', 'swimming_pool/outdoor', 'synagogue/indoor', 'synagogue/outdoor', 'television_studio', 'temple/east_asia', 'temple/south_asia', 'tennis_court/indoor', 'tennis_court/outdoor', 'tent/outdoor', 'theater/indoor_procenium', 'theater/indoor_seats', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'track/outdoor', 'train_railway', 'train_station/platform', 'tree_farm', 'tree_house', 'trench', 'underwater/coral_reef', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'volleyball_court/indoor', 'volleyball_court/outdoor', 'waiting_room', 'warehouse/indoor', 'water_tower', 'waterfall/block', 'waterfall/fan', 'waterfall/plunge', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'wine_cellar/barrel_storage', 'wine_cellar/bottle_storage', 'wrestling_ring/indoor', 'yard', 'youth_hostel']
# Create the label to ID mapping
label2id = {label: idx for idx, label in enumerate(scene_labels_vit)}
# Reverse the mapping to create ID to label mapping
id2label = {idx: label for label, idx in label2id.items()}

# Create a new run
LOCAL_CHACHE_DIR = '/Users/filippomerlo/Desktop/models_cache'

with wandb.init(project="vit-base-patch16-224_SUN397") as run:
    # Pass the name and version of Artifact
    my_model_name = "model-on5m6wmj:v0"
    my_model_artifact = run.use_artifact(my_model_name)

    # Download model weights to a folder and return the path
    model_dir = my_model_artifact.download(LOCAL_CHACHE_DIR)

    # Load your Hugging Face model from that folder
    #  using the same model class
    vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", cache_dir=LOCAL_CHACHE_DIR)
    vit_model = ViTForImageClassification.from_pretrained(
        model_dir,
        num_labels=len(scene_labels_vit),
        id2label=id2label,
        label2id=label2id,
        cache_dir=LOCAL_CHACHE_DIR
    ).to(device)

def classify_scene_vit(image_picture):
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
#%%
import pickle as pkl
import json
import pandas as pd
from pprint import pprint

# Load the object_scene_rel_matrix file
with open("/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/compute_scene_obj_similarity/tf_scores.pkl", "rb") as file:
    object_scene_rel_matrix = pkl.load(file)

# Load the bert object-scene relatedness scores
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/compute_scene_obj_similarity/ade_scenes_bert_similarities.pkl', "rb") as file:
    bert_scene_object_rel_matrix = pkl.load(file)

# Load the size_mean_matrix file
tp_size_mean_path = '/Users/filippomerlo/Desktop/Datasets/sceneREG_data/THINGS/THINGSplus/Metadata/Concept-specific/size_meanRatings.tsv'
size_mean_matrix = pd.read_csv(tp_size_mean_path, sep='\t', engine='python', encoding='utf-8')
things_words = list(size_mean_matrix['Word'])

# Load the llms norms
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/compute_scene_obj_similarity/llama3_8b_instruct_object_scene_norms.pkl', 'rb') as f:
    llama_norms = pkl.load(f)

#%%
# Load the map_coco2ade file
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/mappings/object_map_coco2ade.json', "r") as file:
    map_coco2ade = json.load(file)

# Load the object_map_ade2things file
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/mappings/object_map_ade2things.json', "r") as file:
    map_ade2things = json.load(file)

# Load the map SUN 2 ADE
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/mappings/sun2ade_map.json', "r") as file:
    map_sun2ade = json.load(file)

# Load ade names
path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
with open(path, 'rb') as f:
    ade20k_index = pkl.load(f)
ade20k_object_names = ade20k_index['objectnames']

# Filter objects by remooving the ones that appears in one scene only
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/compute_scene_obj_similarity/tf_scores.pkl', 'rb') as f:
    tf_scores = pkl.load(f)

# Define a function to set each cell to 1 if it has a non-zero value
def set_to_one(x):
    return 1 if x != 0 else 0
# Apply the function to each cell of the DataFrame
tf_scores_ones = tf_scores.applymap(set_to_one)
row_sums = tf_scores_ones.sum(axis=1)
ones_id = [i for i, num in enumerate(row_sums) if num == 1]
object_to_remove = [x for i, x in enumerate(ade20k_object_names) if i in ones_id]
object_to_remove
#%%

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

#%%
id, val = lowest_k(list(bert_scene_object_rel_matrix['airport_terminal']),10)
for i in id:
    print(ade20k_object_names[i])
#%%
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

