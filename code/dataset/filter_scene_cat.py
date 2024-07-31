#%%
from dataset import *
from config_powerpaint import *
from utils_powerpaint import *

def get_scene_predictions(self):
    all_predictions = []
    all_img_paths = []
    c = 0
    
    for img_name in tqdm(list(self.data.keys())):
        try:
            image = self.data[img_name]
            for fix in image['fixations']:
                if fix['condition'] == 'absent':
                    target = None
                    break
                if 'task' in fix.keys():
                    target = fix['task']
                    break
                else:
                    target = None
                
            images_paths = get_files(coco_images_path)
            image_picture = None

            for image_path in images_paths:
                if img_name in image_path:
                    image_picture = Image.open(image_path)
                    all_img_paths.append(image_path)
                    break
            label = classify_scene_vit(image_picture)
            all_predictions.append(label)
        except:
            c += 1
            continue
    count = Counter(all_predictions)
    print(c)
    label_with_paths = dict()
    for i, lab in enumerate(all_predictions):
        if lab not in label_with_paths.keys():
            label_with_paths[lab] = list()
        label_with_paths[lab].append(all_img_paths[i])
    return count, label_with_paths

dataset = Dataset(dataset_path = dataset_path)
count, label_with_paths = get_scene_predictions(dataset)

from pprint import pprint
pprint(count)
