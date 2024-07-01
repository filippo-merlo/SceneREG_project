#%%
import pickle as pkl 
import pandas as pd
import json 
from pprint import pprint
import re

# Load the size_mean_matrix file
size_mean_path = '/Users/filippomerlo/Desktop/Datasets/sceneREG_data/THINGS/THINGSplus/Metadata/Concept-specific/size_meanRatings.tsv'
size_mean_matrix = pd.read_csv(size_mean_path, sep='\t', engine='python', encoding='utf-8')
things_words_id = list(size_mean_matrix['uniqueID'])
things_words_context = list(size_mean_matrix['WordContext'])
things_words_context

# find ade object in things 
path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
# Load the ADE20K index file
with open(path, 'rb') as f:
    ade20k_index = pkl.load(f)

ade20k_object_names = ade20k_index['objectnames']
ade20k_object_names

coco_categories =  [{"supercategory": "person","id": 1,"name": "person"},{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]
coco_objects = []
for d in coco_categories:
    coco_objects.append(d['name'])

#%% LETS EXPLORE NGRAMS!
with open('/Users/filippomerlo/Desktop/Datasets/Natural Language Corpus Data: Beautiful Data/count_1w.txt','r') as f:
    count_Nw = f.read()
    count_Nw_list = count_Nw.split('\n')
    mono_grams_and_f = {}
    mono_grams = []
    mono_grams_f = []
    for couple in count_Nw_list:
        try:
            w, f = couple.split('\t')
            mono_grams.append(w)
            mono_grams_f.append(int(f))
            mono_grams_and_f[w] = f
        except:
            print(couple)
pprint(mono_grams)

#%% MAP and save
from difflib import SequenceMatcher
import inflect

# Initialize the inflect engine
p = inflect.engine()

def normalize_name(name):
    """ Normalize the name by removing special characters and converting to singular form """
    name = name.replace('-', ' ')
    name = p.singular_noun(name) or name
    return name.lower()

from tqdm import tqdm
map_things2ade = {}

brachet_pattern = r"\s*\([^)]*\)"
final_s_pattern = r"s$"

for thing in tqdm(things_words_context):
    
    map_things2ade[thing] = []
    max = 0
    t = normalize_name(re.sub(brachet_pattern, "", thing)) # remove brachets
    ts = t.split()
    try:
        for ade_str in ade20k_object_names:
            ade_list = ade_str.split(', ')
            if ade_list[0] == '-' or ade_list[0] == ' ': # skip
                continue
            l = [normalize_name(x) for x in ade_list] # process
            for t_ in ts:
                for l_ in l:
                    l_nospace = l_.replace(' ','')
                    if re.search(t_, l_nospace): 
                        ratio = SequenceMatcher(None, t, l_).ratio()
                        if ratio > max:
                            max = ratio
                            map_things2ade[thing] = [ade_str]
                        elif ratio == max:
                            map_things2ade[thing].append(ade_str)
                        else:
                            continue
    except:
        print(ade_list)

# remove doubles
for k,v in map_things2ade.items():
    v = list(set(v))
    map_things2ade[k] = v

with open('things2ade_1to1.json', 'w') as f:
    json.dump(map_things2ade, f, indent=4)
#%% REVIEW AND MODIFY

# Check how many things are mapped to more than n ade objects
i = 0
n = 0
things_not_found = []
ade_found = []
for k,v in map_things2ade.items():
    if len(v) > n:
        i+=1
        print(f"'{k}':", v,',')
        things_not_found.append(k)
print(i)

#%%
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
model_name = "bert-base-cased"
device = 'mps'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][:,0,:].squeeze().to('cpu').numpy()

from scipy.spatial.distance import cosine
def cosine_similarity(a, b):
    return 1 - cosine(a, b)

keys_to_review = []
for thing, objects in tqdm(map_things2ade.items()):
    all_objects = []
    for object in objects:
        all_objects += object.split(', ')
    for o  in all_objects:
        if cosine_similarity(get_bert_embedding(thing),
                            get_bert_embedding(o)) <= 0.90:
            keys_to_review.append(thing)

keys_to_review = list(set(keys_to_review))
#%%
pprint(keys_to_review)