#%% IMPORTS
import json
import requests
import pickle as pkl
from pprint import pprint
from config import *  # Import all contents from the config module

#%% LOAD COCO-SEARCH18 DATASET
dataset_path = '/Users/filippomerlo/Desktop/Datasets/data/coco_search18_annotated.json'

# Load the JSON dataset
with open(dataset_path) as f:
    coco_data = json.load(f)

# Extract unique target object names from COCO-SEARCH18 dataset
objects_list = []

for img in coco_data.values():
    annotations = img.get('instances_train2017_annotations') or img.get('instances_val2017_annotations')
    for ann in annotations:
        category_id = ann['category_id']
        for category in coco_categories:
            if category['id'] == category_id:
                object_name = category['name']
                if object_name not in objects_list:
                    objects_list.append(object_name)

print(f'Target Objects, N: {len(objects_list)}')
pprint(objects_list)
print(len(objects_list))
#%% LOAD ADE20K OBJECT NAMES AND IDS
dataset_path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01'
index_file = 'index_ade20k.pkl'

# Load the ADE20K index file
with open(f'{dataset_path}/{index_file}', 'rb') as f:
    ade20k_index = pkl.load(f)

ade20k_object_names = ade20k_index['objectnames']

print(f'ADE Objects, N: {len(ade20k_object_names)}')
pprint(ade20k_object_names)

#%% CHECK IF COCO OBJECTS ARE PRESENT IN ADE20K
objects_found = {}
objects_not_found = []

# Check each object in the COCO objects list
for coco_object in objects_list:
    found = False
    for i, ade20k_object in enumerate(ade20k_object_names):
        ade20k_object_list = ade20k_object.split(', ')
        coco_obj_nospace = coco_object.replace(' ', '')
        ade20k_object_list_nospace = [obj.replace(' ', '') for obj in ade20k_object_list]
        if coco_obj_nospace in ade20k_object_list_nospace:
            objects_found[coco_object] = [i,ade20k_object]
            found = True
            break
    if not found:
        objects_not_found.append(coco_object)

print(f'Objects not found, N: {len(objects_not_found)}')
pprint(objects_not_found)

# For reference, print all ADE20K object names
pprint(objects_found)

#%% MATCH WITH COS SIM
# set device
import torch 
if torch.backends.mps.is_available():
   device = torch.device("mps")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2").to(device)
#%%
ade20k_object_names_2embeed = []
for lis in ade20k_object_names:
    lis = lis.split(', ')
    if len(lis) == 0:
        ade20k_object_names_2embeed.append(lis[0])
    else:
        for name in lis:
            ade20k_object_names_2embeed.append(name)

#Encoding:
with torch.no_grad():
    embeddings_cocoo = model.encode(objects_not_found)
    embeddings_adao = model.encode(ade20k_object_names_2embeed)

# cosine similarity 
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity_matrix = cosine_similarity(embeddings_cocoo, embeddings_adao)
#%%
import numpy as np
# Initialize an empty list to store the indexes
top_5_indexes_per_row = []

# Loop through each row
for row in cosine_similarity_matrix:
    # Get the indexes of the top 5 values in the current row
    top_5_indexes = np.argsort(row)[-10:][::-1]
    # Append the indexes to the list
    top_5_indexes_per_row.append(top_5_indexes)

# Convert the list to a NumPy array if needed
top_5_indexes_per_row = np.array(top_5_indexes_per_row)

# Print the result
for i, idxs in enumerate(top_5_indexes_per_row):
    print(objects_not_found[i])
    print([ade20k_object_names_2embeed[i] for i in idxs])

#%%
# Complete the mapping manualy 
plus = {
    'broccoli':'spinach',
    'wine glass':'drinking glass',
    'potted plant':'potted fern',
    'dining table':'coffee table',
    'carrot':'vegetables',
    'pizza':'bread',
    'donut':'muffin',
    'stop sign':'street sign',
    'banana':'bananas',
    'hair drier':'hair dryer',
    'sports ball':'ball',
    'motorcycle':'motorbike',
    'hot dog':'sandwich'
}

for i, ade20k_object in enumerate(ade20k_object_names):
    lis = ade20k_object.split(', ')
    for cocon, aden in plus.items():
        if aden in lis:
            objects_found[cocon] = [i,ade20k_object]
pprint(objects_found)
#%%
pprint(objects_found)
print(len(objects_found.keys()))
#%%
#import json 
#with open('object_map_coco2ade.json', 'w') as f:
#    json.dump(objects_found, f, indent=4)

#%%

# Labels and occurrencies
with open('object_scenes_cooccurrency.pkl', 'rb') as f:
    object_scenes_cooccurrency = pkl.load(f)

# %%
object_scenes_cooccurrency
len(objects_found.keys())
#%%
# Get the n most related scene category for each object
k = 100
most_related_scene_cat = []

for id in objects_found.values():
    #print(id[1])
    id = id[0]
    # Get the nlargest values along with their indices
    largest_elements = object_scenes_cooccurrency.iloc[id].nlargest(k)
    # Separate the indices and values
    cats = largest_elements.index
    values = largest_elements.values
    values = [1 if v > 0 else 0 for v in values]
    cats = [cat for cat, v in zip(cats, values) if v == 1]
    most_related_scene_cat += cats

from collections import Counter
counter = Counter(most_related_scene_cat)
pprint(counter)
print(len(set(most_related_scene_cat)))

to_keep = [k for k,v in counter.items() if v > 10]
if 'outlier' in to_keep:
    to_keep.remove('outlier')
print(to_keep)
print(len(to_keep))

#%%
to_keep = ['bathroom', 'bedroom', 'hotel_room', 'game_room', 'living_room', 'office', 'nursery', 'restaurant', 'dining_room', 'kitchen', 'attic', 'galley', 'wet_bar', 'kitchenette', 'vehicle', 'dinette_home', 'poolroom_home', 'conference_room', 'closet', 'bar', 'bow_window_indoor', 'basement', 'art_gallery', 'classroom', 'corridor', 'youth_hostel', 'coffee_shop', 'library_indoor', 'kindergarden_classroom', 'recreation_room', 'dorm_room', 'childs_room', 'artists_loft', 'home_office', 'art_studio', 'highway', 'dining_hall', 'street', 'restaurant_patio', 'lobby', 'waiting_room', 'dining_car', 'reception', 'parlor', 'shop', 'airplane_cabin', 'conference_center', 'airport_terminal', 'pantry', 'plaza', 'building_facade']