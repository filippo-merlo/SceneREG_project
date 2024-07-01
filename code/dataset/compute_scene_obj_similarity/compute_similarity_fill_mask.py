#%% COUMPUTE SIMILARITY WITH BERT
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn.functional import softmax
import pandas as pd
import pickle as pkl
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

CACHE_DIR = '/mnt/cimec-storage6/users/filippo.merlo'
# Load the pre-trained model and tokenizer
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = CACHE_DIR)
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir = CACHE_DIR).to(device)

# get objects and scenes names 
# Load index with global information about ADE20K
DATASET_PATH = '/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

candidates = index_ade20k['objectnames']
from datasets import load_dataset
ade_hf_data = load_dataset("scene_parse_150", cache_dir=CACHE_DIR)
scenes_categories = ade_hf_data['train'].features['scene_category'].names

# Function to calculate the probability of a candidate
def get_candidate_probability(candidate_tokens):

    # Replace the masked token with the candidate tokens
    tokenized_candidate = ["[CLS]"] + tokenized_text[:mask_token_index] + candidate_tokens + tokenized_text[mask_token_index + 1:]

    # Convert tokenized sentence to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_candidate)

    # Convert input IDs to tensors
    input_tensor = torch.tensor([input_ids]).to(device)

    # Get the logits from the model
    with torch.no_grad():
        logits = model(input_tensor).logits[0].to('cpu')

    # Calculate the probability of the candidate word
    probs = softmax(logits, dim=-1)
    probs = probs[range(len(input_ids)), input_ids]
    prob = (
        torch.prod(probs[1:mask_token_index+1])
        * torch.prod(probs[mask_token_index+len(candidate_tokens)+1:])
    )

    return prob.item()

def name2idx(name, name_list):
    return name_list.index(name)

bert_similarities_mat = pd.DataFrame(columns=scenes_categories, index=range(len(index_ade20k['objectnames'])))

for scene in tqdm(scenes_categories):
    # Define the input sentence with a masked word
    input_text = "In the [SCENE] there is a [MASK]."
    input_text = input_text.replace("[SCENE]", scene.replace('_', ' '))
    print(input_text)
    # Tokenize the input sentence
    tokenized_text = tokenizer.tokenize(input_text)
    mask_token_index = tokenized_text.index("[MASK]")
    
    # Evaluate the probability of each candidate word
    for candidate in candidates:
        candidate_tokens = tokenizer.tokenize(candidate)
        candidate_probability = get_candidate_probability(candidate_tokens)
        bert_similarities_mat.loc[name2idx(candidate, candidates), scene] = candidate_probability

bert_similarities_mat.to_pickle('{}/{}'.format(CACHE_DIR, "ade_scenes_bert_similarities.pkl"))

#%%
# Author: Shivika Sharma
# 27 November 2023

import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers import BertTokenizer
import random

class My_BERT(nn.Module):
 def __init__(self, model_name, tokenizer, answer_tokens, c_tokens=None, is_map=False):
  super(My_BERT, self).__init__()
  self.BERT = BertForMaskedLM.from_pretrained(model_name)
  self.tokenizer = tokenizer
  for param in self.BERT.parameters():
   param.requires_grad = True

  self.answer_ids = self.tokenizer.encode(answer_tokens, add_special_tokens=False)
  self.N = len(answer_tokens)
  self.mask_token_id = 103
  self.loss_func = nn.CrossEntropyLoss()
  self.is_map = False
  if is_map:
   self.class_tokens = [self.tokenizer.encode(c_tk, add_special_tokens=False) for c_tk in c_tokens]
   self.is_map = True

 def forward(self, input_id, input_label):
  input_label = torch.tensor([input_label])
  outputs = self.BERT(input_ids=input_id['input_ids'],attention_mask=input_id['attention_mask'])
  out_logits = outputs.logits

  mask_position = input_id['input_ids'].eq(self.mask_token_id)
  mask_logits = out_logits[mask_position, :].view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]

  answer_logits = mask_logits[:, self.answer_ids]
  if self.is_map:
   for c_index in range(len(self.class_tokens)):
    answer_logits[0][c_index] = torch.sum(mask_logits[:, self.class_tokens[c_index]])

  answer_probs = answer_logits.softmax(dim=1)

  loss = self.loss_func(answer_logits, input_label)

  return loss, answer_probs

 def return_probs(self, input_id):
  _, answer_probs = self.forward(input_id, random.randint(0,self.N-1))
  token_probs = zip(self.answer_ids, answer_probs[0])
  token_probs = [(self.tokenizer.convert_ids_to_tokens(ans_tk),ans_prob.item()) for ans_tk,ans_prob in token_probs]

  return sorted(token_probs, key=lambda x: x[1], reverse=True)

 def return_prediction(self, input_id):
  _, answer_probs = self.forward(input_id, random.randint(0,self.N-1))
  token_probs = zip(self.answer_ids, answer_probs[0])
  return self.tokenizer.convert_ids_to_tokens(max(token_probs, key=lambda x: x[1])[0])

def custom_model(model_name, answer_tokens, c_tokens=None):
 tokenizer = BertTokenizer.from_pretrained(model_name)

 if c_tokens is not None:
  is_map = True
 else:
  is_map = False
 model = My_BERT(model_name, tokenizer, answer_tokens, c_tokens=c_tokens, is_map=is_map)

 return model, tokenizer

from pprint import pprint
input = 'There is a [MASK] in the airport terminal'
candidates = index_ade20k['objectnames']
pprint(candidates)