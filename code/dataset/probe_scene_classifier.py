#%%
from config import * 
from utils import *
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel

cache_dir = '/mnt/cimec-storage6/users/filippo.merlo'
ds = load_dataset("scene_parse_150", cache_dir= cache_dir)

# Remove test split
dataset = DatasetDict()
dataset = concatenate_datasets([ds['train'], ds['validation']])
to_keep = ['bathroom', 'bedroom', 'hotel_room', 'game_room', 'living_room', 'office', 'nursery', 'restaurant', 'dining_room', 'kitchen', 'attic', 'galley', 'wet_bar', 'kitchenette', 'vehicle', 'dinette_home', 'poolroom_home', 'conference_room', 'closet', 'bar', 'bow_window_indoor', 'basement', 'art_gallery', 'classroom', 'corridor', 'youth_hostel', 'coffee_shop', 'library_indoor', 'kindergarden_classroom', 'recreation_room', 'dorm_room', 'childs_room', 'artists_loft', 'home_office', 'art_studio', 'highway', 'dining_hall', 'street', 'restaurant_patio', 'lobby', 'waiting_room', 'dining_car', 'reception', 'parlor', 'shop', 'airplane_cabin', 'conference_center', 'airport_terminal', 'pantry', 'plaza', 'building_facade']
for label in to_keep:
    if label[0].lower() in vowels:
        scene_labels_context.append(f"a picture of an {label.replace('_', ' ')}")
    else:
        scene_labels_context.append(f"a picture of a {label.replace('_', ' ')}")

# Remove misc
scene_names = list(dataset.features['scene_category'].names)
names2id = dict(zip(scene_names, range(len(scene_names))))
names2id_filtered = dict()

for label in scene_names:
    #if label == 'misc':
    #    continue
    if label not in to_keep:
        continue
    else:
        names2id_filtered[label] = names2id[label]
filter_dataset = dataset.filter(lambda example: example['scene_category'] in names2id_filtered.values())

filter_dataset['scene_category']
names2id_filtered
new_scene_categories = list(names2id_filtered.keys())

mapOldid2Newid = {}
for i,id in enumerate(names2id_filtered.values()):
    mapOldid2Newid[id] = i

new_label_ids = [mapOldid2Newid[id] for id in filter_dataset['scene_category']]
final_dataset = filter_dataset.remove_columns('scene_category').add_column('scene_category', new_label_ids)

# Redefine class labels
class_labels = ClassLabel(names=new_scene_categories, num_classes=len(new_scene_categories))
final_dataset =  final_dataset.cast_column('scene_category', class_labels)
final_dataset = final_dataset.train_test_split(test_size=0.2)
new_names2id = dict(zip(new_scene_categories,range(len(new_scene_categories))))
