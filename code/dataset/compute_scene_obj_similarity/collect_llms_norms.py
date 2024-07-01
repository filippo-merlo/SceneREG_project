#%%
device = "cuda:0"
CACHE_DIR = '/mnt/cimec-storage6/shared'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ACCESS_TOKEN = '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'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
dtype = torch.bfloat16

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    cache_dir=CACHE_DIR, 
    token=ACCESS_TOKEN
)

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

yes_token = tokenizer('YES', return_tensors="pt", add_special_tokens=False).input_ids
no_token = tokenizer('NO', return_tensors="pt", add_special_tokens=False).input_ids

from tqdm import tqdm

answers = {}
for scene_name in tqdm(scenes_categories):
    answers[scene_name] = {}
    candidate_scores = []
    for candidate in candidates:
        candidate_list = candidate.split(', ')
        answers[scene_name][candidate] = []

        for single_candidate in candidate_list:
            prompt = "Is it possible to find the object '{word}' in the place '{scene}'?".format(word=single_candidate, scene=scene_name.replace("_", " "))
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Your job is to say if an object can be found in a specific place. You can answer only with YES or NO."},
                {"role": "user", "content": prompt}
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            
            with torch.no_grad():
                distribution = model(input_ids=input_ids)

            probs = torch.nn.functional.softmax(distribution.logits, dim=-1).to('cpu')
            yes_prob = probs[0, -1, yes_token].squeeze().item()
            no_prob = probs[0, -1, no_token].squeeze().item()

            # Add answer and probabilities to the list
            answers[scene_name][candidate].append([single_candidate, yes_prob, no_prob])

with open('llama3_8b_instruct_object_scene_norms.pkl', 'wb') as f:
    pkl.dump(answers, f)
