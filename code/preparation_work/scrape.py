#%%
import requests
import json 
link = 'http://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json'
json_response = requests.get(link)

# Step 5: Parse the JSON data
json_data = json_response.json()


file_path = "/Users/filippomerlo/Documents/GitHub/MakingAScene/data/coco_search18_TP/COCOFreeView_fixations_trainval.json"

# Write data to JSON file
with open(file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)