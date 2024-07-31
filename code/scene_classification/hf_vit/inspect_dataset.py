#%%
### PREPARE THE DATASET  
import sys
sys.path.append('../')
from config import *
# Load the dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
ds = load_dataset("scene_parse_150", cache_dir= cache_dir)

# Remove test split
dataset = DatasetDict()
dataset = concatenate_datasets([ds['train'], ds['validation']])

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


import json
with open('/home/filippo.merlo/SceneREG_project/code/scene_classification/finetuning/hf_vit/new_labels.json', 'r') as f:
    new_labels = json.load(f)

new_scene_categories = [new_labels['scene_labels'][i] for i in new_labels['scene_ids']]

new_label_ids = new_labels['img_label_ass']

final_dataset = filter_dataset.remove_columns('scene_category').add_column('scene_category', new_label_ids)

# Redefine class labels
class_labels = ClassLabel(names=new_scene_categories, num_classes=len(new_scene_categories))
final_dataset =  final_dataset.cast_column('scene_category', class_labels)


names = final_dataset.features['scene_category'].names
id2names = dict(zip(range(len(names)), names))

# Count the occurrences of each label
from collections import Counter
tot_labs = final_dataset['scene_category']
counter = Counter(tot_labs)
print(len(counter.keys()))
# Get the labels
labels = list(counter.keys())
names2id_filtered = dict()

for l in labels:
    if counter[l] > 10:
        print(l,':',counter[l])

#%%
#final_dataset = final_dataset.train_test_split(test_size=0.1)
#print(len(set(final_dataset['train']['scene_category'])))
#print(len(set(final_dataset['test']['scene_category'])))
#new_names2id = dict(zip(new_scene_categories,range(len(new_scene_categories))))