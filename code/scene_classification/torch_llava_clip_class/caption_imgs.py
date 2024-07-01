#%%
from dataset import *
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from dataset_prep import final_dataset

# works with final dataset with tresh 30 and train-test split at 0.1

# Initialize Weights and Biases (wandb)
cache_dir = '/mnt/cimec-storage6/users/filippo.merlo'
import os
os.environ['HF_HUB_CACHE'] = cache_dir

#%%
# Specify Device (GPU/CPU)
import torch

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Initialize DataLoader and Preprocessor
from transformers import AutoProcessor, CLIPModel, LlavaForConditionalGeneration, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # or load_in_8bit=True for 8-bit quantization
    bnb_4bit_compute_dtype=torch.float16  # specify compute dtype
)

clip =CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device0)
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
llava = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-13b-hf", 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2",
                cache_dir=cache_dir+'/hub/'
            )

prompt = "USER: <image>\nWhere is the picture taken?\nASSISTANT:"

def getitem(idx, data):
    image = data[idx]['image']
    # process image and text
    with torch.no_grad():
        llava_inputs = llava_processor(prompt, image, return_tensors='pt').to(device1, torch.float16)
        llava_encode = llava.generate(**llava_inputs, max_new_tokens=75, do_sample=False)
        llava_caption = llava_processor.decode(llava_encode[0][2:], skip_special_tokens=True)
        inputs = clip_processor(text=str(llava_caption), images=image, return_tensors="pt", padding=True).to(device0)
        outputs = clip(**inputs)
        #txt_features = outputs.text_model_output.last_hidden_state.mean(dim=1) 
        #img_features = outputs.vision_model_output.last_hidden_state.mean(dim=1) 
        txt_features = outputs.text_model_output.text_embeds()
        img_features = outputs.vision_model_output.image_embeds()
        reppresentation = torch.cat([txt_features, img_features], dim=1).squeeze()

    return reppresentation

import pickle 
def save_append_list(path, list):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    data += list
    print(len(data))
    with open(path, 'wb') as f:
        pickle.dump(data, f)

from tqdm import tqdm
def embed_data(data,path):
    len_data = len(data)
    r_list = []
    with open(path, 'wb') as f:
        pickle.dump(r_list, f)

    for i in tqdm(range(len_data)):
        r_list.append(getitem(i, data))
        if i % 100 == 0:
            save_append_list(path, r_list)
            r_list = []
    save_append_list(path, r_list)


data_tr = final_dataset['train']
data_te = final_dataset['test']

embed_data(data_tr, cache_dir+'/'+'train_rep.pkl')

embed_data(data_te, cache_dir+'/'+'test_rep.pkl')



