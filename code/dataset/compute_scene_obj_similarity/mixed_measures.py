#%%
import pickle as pkl 
import pandas as pd
import json 
#%%
# Load the object_scene_rel_matrix file
with open("/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/compute_scene_obj_similarity/tf_scores.pkl", "rb") as file:
    object_scene_rel_matrix = pkl.load(file)
object_scene_rel_matrix
#%%
# Load the size_mean_matrix file
size_mean_path = '/Users/filippomerlo/Desktop/Datasets/sceneREG_data/THINGS/THINGSplus/Metadata/Concept-specific/size_meanRatings.tsv'
size_mean_matrix = pd.read_csv(size_mean_path, sep='\t', engine='python', encoding='utf-8')
things_words = list(size_mean_matrix['Word'])
len(things_words)
#%%
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