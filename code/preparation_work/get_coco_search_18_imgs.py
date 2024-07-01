#%% Get COCO-Search-18 image names

### FUNCTIONS ###
import os
import json
from pprint import pprint
import requests

# import json files from a folder
def import_json_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter only JSON files
    json_files = [f for f in files if f.endswith('.json')]
    
    # Initialize an empty list to store loaded JSON data
    json_data = []
    
    # Iterate over each JSON file
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            try:
                # Load JSON data from the file
                data = json.load(file)
                json_data.append(data)
            except Exception as e:
                print(f"Error loading JSON from {file_name}: {e}")
    
    return json_data

def save_image_from_url(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print("Image saved successfully.")
        else:
            print("Failed to download image. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred:", str(e))

#%%
### EXECUTION ###

# import all JSON files from the folder
folder_path = '/Users/filippomerlo/Desktop/Datasets/data/coco_search18_TP'
all_sets = import_json_files(folder_path)

image_ids = []
for set in all_sets:
    for trial in set:
        if trial['name'] not in image_ids:
            image_ids.append(trial['name'])
print(len(image_ids))


#%%
from pycocotools.coco import COCO
from PIL import Image

imgs = []
dataType=['train2017','val2017']
save_path = '/Users/filippomerlo/Desktop/Datasets/data/images/'
for path in dataType:
    annotations_path = f'/Users/filippomerlo/Desktop/Datasets/data/coco_annotations/annotations/instances_{path}.json'
    with open(annotations_path) as f:
        annotations = json.load(f)
        for i in annotations['images']:
            if i['file_name'] in image_ids:
                save_image_from_url(i['coco_url'],save_path+i['file_name'])
