import pickle as pkl
import json
import re
import itertools
import pandas as pd
# find ade object in things 
path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
# Load the ADE20K index file
with open(path, 'rb') as f:
    ade20k_index = pkl.load(f)
ade20k_object_names = ade20k_index['objectnames']

# Load the size_mean_matrix file
tp_size_mean_path = '/Users/filippomerlo/Desktop/Datasets/osfstorage-archive/THINGSplus/Metadata/Concept-specific/size_meanRatings.tsv'
size_mean_matrix = pd.read_csv(tp_size_mean_path, sep='\t')
things_words = list(size_mean_matrix['Word'])

import re
mapping_ade2things = {}
for ade_names in ade20k_object_names:
    ade_name_list = ade_names.lower().replace(' ','').replace('-','').split(',')
    for ade_name in ade_name_list:
        for things_name in things_words:
            if re.sub(r's$', '',ade_name) == re.sub(r's$', '',things_name.replace(' ','').replace('-','')):
                if ade_names not in mapping_ade2things:
                    mapping_ade2things[ade_names] = [things_name]
                else:
                    mapping_ade2things[ade_names].append(things_name)


# coco 63 obj --> ade20k
#%%
# Load the map_coco2ade file
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/object_scene_rel/object_map_coco2ade.json', "r") as file:
    map_coco2ade = json.load(file)

ade20k_coco_o = set([object_name[1] for object_name in map_coco2ade.values()])
len(ade20k_coco_o)

#%%
# ade20k --> things
mapping_ade2things
ade20k_things_o = set(list(mapping_ade2things.keys()))
len(ade20k_things_o)

#%%

ade20k_coco_o - ade20k_things_o # = {'tennis racket'}

# manually add tennis racket
mapping_ade2things['tennis racket'] = ['racket']
mapping_ade2things['minibike, motorbike'] = ['motorcycle']
mapping_ade2things['potted fern'] = ['plate']
mapping_ade2things['street sign'] = ['road sign']
with open('object_map_ade2things.json', 'w') as f:
    json.dump(mapping_ade2things, f, indent=4)
