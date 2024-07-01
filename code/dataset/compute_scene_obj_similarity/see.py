#%%
import pickle as pkl

with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/compute_scene_obj_similarity /llama3_8b_instruct_object_scene_norms.pkl', 'rb') as f:
    llama_norms = pkl.load(f)

#%%
from pprint import pprint
pprint(llama_norms['airport_terminal'])

#%%
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/compute_scene_obj_similarity /tf_scores.pkl', 'rb') as f:
    tf_scores = pkl.load(f)
# Define a function to set each cell to 1 if it has a non-zero value
def set_to_one(x):
    return 1 if x != 0 else 0
# Apply the function to each cell of the DataFrame
tf_scores_ones = tf_scores.applymap(set_to_one)
row_sums = tf_scores_ones.sum(axis=1)
ones_id = [i for i, num in enumerate(row_sums) if num == 1]
len(ones_id)
#%%
from collections import Counter
frequency = Counter(row_sums)

#%%
frequency

#%%
path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
with open(path, 'rb') as f:
    ade20k_index = pkl.load(f)
ade20k_object_names = ade20k_index['objectnames']
pprint(ade20k_object_names)