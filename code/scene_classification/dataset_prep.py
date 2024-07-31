#%%
### PREPARE THE DATASET   
from config import *
import random
random.seed(42)

# Load the dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
ds = load_dataset("scene_parse_150", cache_dir=cache_dir)

# Remove test split
dataset = DatasetDict()
dataset = concatenate_datasets([ds['train'], ds['validation']])

# Remove misc
#scene_names = list(dataset.features['scene_category'].names)
#names2id = dict(zip(scene_names, range(len(scene_names))))
#names2id_filtered = dict()
#
#to_keep = ['bathroom', 'bedroom', 'game_room', 'living_room', 'office',
#           'restaurant', 'dining_room', 'kitchen', 'attic',
#           'vehicle', 'closet', 'bar', 
#          'basement', 'corridor',
#           'coffee_shop', 'library_indoor',
#           'home_office', 'art_studio', 'highway',
#           'street',
#           'shop']
#
#for label in scene_names:
#    #if label == 'misc':
#    #    continue
#    if label not in to_keep:
#        continue
#    else:
#        names2id_filtered[label] = names2id[label]
#filter_dataset = dataset.filter(lambda example: example['scene_category'] in names2id_filtered.values())

# ALREADY DONE; JUST IMPORT THE DICT WITH NEW LABLES
'''
### CLUSTER LABELS
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch

# cuda 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir).to(device)
#txt_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir).to(device)
auto_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir)


# Remove misc
scene_names = list(dataset.features['scene_category'].names)
names2id = dict(zip(scene_names, range(len(scene_names))))
names2id_filtered = dict()
for label in scene_names:
    if label == 'misc':
        continue
    else:
        names2id_filtered[label] = names2id[label]
filter_dataset = dataset.filter(lambda example: example['scene_category'] in names2id_filtered.values())


from tqdm import tqdm
import numpy as np 
data_points = []
captions = dict()

with torch.no_grad():
    for c_l in scene_names:
        txt_inputs = tokenizer(f'the picture of a {c_l.replace('_', ' ')}', return_tensors="pt").to(device)
        captions[c_l] = clip_model.get_text_features(**txt_inputs).to('cpu')

    # preprocess and embed imgs and labels
    for i in tqdm(range(len(filter_dataset))):
        v_inputs = auto_processor(images=filter_dataset[i]['image'], return_tensors="pt").to(device)
        image_embeds = clip_model.get_image_features(**v_inputs).to('cpu')
        data_points.append(image_embeds)

    data_points = torch.stack(data_points).squeeze().detach().numpy()

from sklearn import cluster
# ---------- K-Mean clustering simplified ----------
k = 100
clusters = cluster.KMeans(n_clusters=k).fit(data_points)
#print(clusters.cluster_centers_.shape) # here there are the centroids (k, 768)
img_label_ass =  clusters.labels_
scene_labels = list(captions.keys())
labels_emb = torch.stack(list(captions.values())).squeeze().detach().numpy()
# find the labels most similar to the centroids
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(clusters.cluster_centers_, labels_emb)
idxs = np.argmax(cosine_sim, axis=1)
idxs_tk = np.argsort(cosine_sim, axis=1)[:,-1054:]
print(idxs_tk)
# Handle duplicate assignments (replace with actual uniqueness check and refinement logic)

def remove_dup(idxs, idxs_tk, k):
    id_record = dict()
    while True:
        unique_idxs = set(idxs)
        if len(unique_idxs) >= k:
            break

        for i in range(len(idxs)):
            if np.count_nonzero(idxs == idxs[i]) > 1:  # Check for duplicates
                if str(i) not in id_record.keys():
                    id_record[str(i)] = 0
                else:
                    id_record[str(i)] += 1

                # Ensure we do not go out of bounds
                if id_record[str(i)] < len(idxs_tk[i]):
                    idxs[i] = idxs_tk[i][id_record[str(i)]]
                else:
                    raise ValueError(f"No more unique values available for index {i}")

        unique_idxs = set(idxs)
        print(len(unique_idxs))
        if len(unique_idxs) >= k:
            break

    return idxs

idxs = remove_dup(idxs, idxs_tk, k)

# save the labels
new_labels = {
    'scene_labels' : list(scene_labels),
    'scene_ids' : [int(i) for i in idxs],
    'img_label_ass' : [int(i) for i in img_label_ass]
}

# save new_labels dict in json format
import json 
with open('new_labels.json', 'w') as f:
    json.dump(new_labels, f)
'''
#import json
#with open('/home/filippo.merlo/SceneREG_project/code/scene_classification/finetuning/hf_vit/new_labels.json', 'r') as f:
#    new_labels = json.load(f)
#
#new_scene_categories = [new_labels['scene_labels'][i] for i in new_labels['scene_ids']]
#
#new_label_ids = new_labels['img_label_ass']
#
#final_dataset = filter_dataset.remove_columns('scene_category').add_column('scene_category', new_label_ids)

'''
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
print(len(set(final_dataset['train']['scene_category'])))
print(len(set(final_dataset['test']['scene_category'])))

'''
### FILTER LABELS

# Inspect the dataset and counting the number of occurrences of each label 'name'
from collections import Counter
import json

names = dataset.features['scene_category'].names
id2names = dict(zip(range(len(names)), names))

#%%
# Count the occurrences of each label
tot_labs = dataset['scene_category']
counter = Counter(tot_labs)

THRESHOLD_CLASSES = 30
from pprint import pprint

#pprint({names[k]:v for k, v in counter.items() if v >= THRESHOLD_CLASSES})

# Get the labels
labels = list(counter.keys())
names2id_filtered = dict()

for label in labels:
    if counter[label] >= THRESHOLD_CLASSES:
        names2id_filtered[id2names[label]] = label

#pprint(names2id_filtered.keys())
#len(names2id_filtered.keys())
filter_dataset = dataset.filter(lambda example: example['scene_category'] in names2id_filtered.values())

# make dicts
new_names2id = dict()

for i, name in enumerate(names2id_filtered.keys()):
    new_names2id[name] = i

# reverse dict
id2names = {v: k for k, v in new_names2id.items()}
old_2_new_map = dict()

for name, old_id in names2id_filtered.items():
    new_id = new_names2id[name]
    old_2_new_map[old_id] = new_id

### ADJUST LABELS

# map old labels to new labels
new_labels= [old_2_new_map[x] for x in filter_dataset['scene_category']]
final_dataset = filter_dataset.remove_columns('scene_category').add_column('scene_category', new_labels)

# Redefine class labels
class_labels = ClassLabel(names=list(names2id_filtered.keys()), num_classes=len(names2id_filtered.keys()))
final_dataset =  final_dataset.cast_column('scene_category', class_labels)

final_dataset = final_dataset.train_test_split(test_size=0.1)