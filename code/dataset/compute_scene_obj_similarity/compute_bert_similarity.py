import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
CACHE_DIR = '/mnt/cimec-storage6/users/filippo.merlo'
model_name = "bert-large-cased"
device='cuda:2'
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
model = BertModel.from_pretrained(model_name, cache_dir=CACHE_DIR).to(device)


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][:,0,:].squeeze().to('cpu').numpy()

from scipy.spatial.distance import cosine
def cosine_similarity(a, b):
    return 1 - cosine(a, b)

import pickle as pkl
# get objects and scenes names 

# Load object categories from ADE20K 
DATASET_PATH = '/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)
candidates = index_ade20k['objectnames']

# Load scene categories from ADE20K hf
from datasets import load_dataset
ade_hf_data = load_dataset("scene_parse_150", cache_dir='/mnt/cimec-storage6/shared/hf_datasets')
scenes_categories = ade_hf_data['train'].features['scene_category'].names

import pandas as pd
import numpy as np 
 
n_rows = len(candidates)
similarity_matrix = pd.DataFrame(0, index=range(n_rows), columns=scenes_categories)

#for scene in scenes_categories:
airport_scores = []
for scene in ['airport_terminal']:
    for candidate in candidates:
        candidate_list = candidate.split(', ')
        scores = []
        for i, object in enumerate(candidate_list):
            candidate_embedding = get_bert_embedding(object)
            scene_embedding = get_bert_embedding(scene.replace('_', ' '))
            similarity = cosine_similarity(candidate_embedding, scene_embedding)
            scores.append(similarity)
        airport_scores.append(np.mean(scores))
        #similarity_matrix.loc[i, scene] = np.mean(scores)

def get_indexes_top_n(lst, n):
    if n <= 0:
        return []
    sorted_indexes = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
    return sorted_indexes[:n]

top_100 = get_indexes_top_n(airport_scores, 100)
print('airport_terminal:')
for i in top_100:
    print(f'{i}. ',candidates[i], airport_scores[i])