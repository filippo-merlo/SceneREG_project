#%% 
import json 
import os
from pprint import pprint 

# Load current main dataset
with open('/Users/filippomerlo/Desktop/Datasets/data/coco_search18_annotated.json', 'r') as f:
    main_data = json.load(f)
i = 0
yes = 0
path_hl = '/Users/filippomerlo/Desktop/Datasets/HL Dataset'
# Iterate through files in the folder
for filename in os.listdir(path_hl):
  # Check if it's a JSON file
  if filename.endswith(".jsonl"):
    # Construct the full path
    file_path = os.path.join(path_hl, filename)

    # Open the file in read mode with proper encoding
    with open(file_path, "r", encoding="utf-8") as f:
      for line in f:
        # Read the content
        image_data = json.loads(line)
        image_number = str(image_data['file_name'][-16:])
        print(image_number)
        try:
            main_data[image_number]['hl_captions'] = image_data["captions"]
            yes += 1
            print('yes',yes)
        except:
           i += 1
           print('no',i)





      

        