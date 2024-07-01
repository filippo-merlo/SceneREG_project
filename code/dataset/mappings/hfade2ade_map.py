#%%
# load ade20k
import pickle as pkl 
ade_path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
# load index
with open(ade_path, 'rb') as f:
    index_ade20k = pkl.load(f)

scene_list = set(index_ade20k['scene'])

from datasets import load_dataset
from PIL import Image

ds = load_dataset("scene_parse_150")
scene_category = ds['train'].features['scene_category'].names

# Map scene list to scene category 
import itertools
import re

def map_scene_to_category(scene_list, scene_category, accurate=False):
    category_to_scene = {}
    for scene in scene_list:
        if accurate:
            scene_l = re.split('/', scene)
            scene_l_clean = []
            for s in scene_l:
                if s[:2] != 'n_' and s[:2] != 'a_':
                    scene_l_clean.append(s)
            scene_l = '_'.join(scene_l_clean)
            scene_l = re.split('_', scene_l)
            
        else:
            scene_l = re.split('/', scene)
        scene_l = [s for s in scene_l if s]  # Remove empty strings
        scene_l = remove_duplicates(scene_l)
        combinations = []
        for r in range(1, len(scene_l) + 1):
            combinations.extend(list(itertools.combinations(scene_l, r)))

        all_permutations_of_combinations = []
        for combo in combinations:
            permutations_of_combo = list(itertools.permutations(combo))
            all_permutations_of_combinations.extend(permutations_of_combo)
        for scene_cat in scene_category:
            for perm in all_permutations_of_combinations:
                perm_str = '_'.join(perm)
                if perm_str == scene_cat:  # Changed from 'scene_category'
                    if scene_cat in category_to_scene.keys():
                        category_to_scene[scene_cat].append(scene)
                    else:
                        category_to_scene[scene_cat] = [scene]
    return category_to_scene

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def reverse_dict(dictionary):
    reversed_dict = {}
    for key, values in dictionary.items():
        for value in values:
            if value not in reversed_dict:
                reversed_dict[value] = [key]
            else:
                reversed_dict[value].append(key)
    return reversed_dict

#%%
category_to_scene = map_scene_to_category(scene_list, scene_category)
scene_to_category = reverse_dict(category_to_scene)
# First check
print(len(set(scene_category)),len(set(scene_category) - set(category_to_scene.keys())))
print(len(scene_list),len(scene_list - set(scene_to_category.keys())))

unmatched_scenes = scene_list - set(scene_to_category.keys())
unmatched_scenes = list(unmatched_scenes)
accurate_category_to_scene = map_scene_to_category(unmatched_scenes, scene_category, accurate=True)
accurate_scene_to_category = reverse_dict(accurate_category_to_scene)

for k, v in accurate_category_to_scene.items():
    category_to_scene[k] += v
for k, v in accurate_scene_to_category.items():
    if k not in scene_to_category:
        scene_to_category[k] = v
    else:
        scene_to_category[k] += v

# Correct ambiguity, mapping in scene_to_category has to be one to one
print('Pre corrections:')
print(len(set(scene_category)),len(set(scene_category) - set(category_to_scene.keys())))
print(len(scene_list),len(scene_list - set(scene_to_category.keys())))
for k, v in scene_to_category.items():
    if len(v) > 1:
        print(k, v)
# copy corrections
corrections = {
'/library/outdoor': ['library_outdoor'],
'library/indoor': ['library_indoor'],
'/library/indoor': ['library_indoor'],
'library/outdoor': ['library_outdoor'],
'utliers/artists_loft/questionable': ['artists_loft'],
'booth/indoor': ['booth_indoor'],
'/booth/indoor': ['booth_indoor'],
'/airport/entrance': ['airport'],
'/cargo_deck/boat': ['boat'],
'/cargo_deck/airplane': ['airplane'],
'/train_station_platform/n_people_waiting_in_a_queue_to_enter_the_train': ['station'],
'/train_station_platform/n_passengers_leaving_a_train': ['station']
}

for k, v in scene_to_category.items():
    if k in corrections.keys():
        scene_to_category[k] = corrections[k]

print('After corrections:')
for k, v in scene_to_category.items():
    if len(v) > 1:
        print(k, v)

category_to_scene = reverse_dict(scene_to_category)

# Final Check
print(len(set(scene_category)),len(set(scene_category) - set(category_to_scene.keys())))
print(len(scene_list),len(scene_list - set(scene_to_category.keys())))

import json

with open('category_to_scene.json', 'w') as f:
    json.dump(category_to_scene, f, indent=4)

with open('scene_to_category.json', 'w') as f:
    json.dump(scene_to_category, f, indent=4)
