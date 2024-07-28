#%%
from config_powerpaint import *
from utils import *
import json
from tqdm import tqdm

class Dataset:

    def __init__(self, dataset_path = None):
        if dataset_path:
            with open(dataset_path) as f:
                self.data = json.load(f)
                self.img_names = list(self.data.keys())
                '''
                for each img name there are three keys:
                Key: [fixations]
                    Key: [name], Type of Value: [str]
                    Key: [subject], Type of Value: [int]
                    Key: [condition], Type of Value: [str]
                    Key: [X], Type of Value: [list]
                    Key: [Y], Type of Value: [list]
                    Key: [T], Type of Value: [list]
                    Key: [length], Type of Value: [int]
                    Key: [split], Type of Value: [str]
                    Key: [fixOnTarget], Type of Value: [bool]
                    Key: [correct], Type of Value: [int]
                Key: [captions_val2017_annotations] or [captions_train2017_annotations]
                    Key: [image_id], Type of Value: [int]
                    Key: [id], Type of Value: [int]
                    Key: [caption], Type of Value: [str]
                Key: [instances_val2017_annotations] or [instances_train2017_annotations]
                    Key: [segmentation], Type of Value: [list]
                    Key: [area], Type of Value: [float]
                    Key: [iscrowd], Type of Value: [int]
                    Key: [image_id], Type of Value: [int]
                    Key: [bbox], Type of Value: [list]
                    Key: [category_id], Type of Value: [int]
                    Key: [id], Type of Value: [int]
                '''
        else:
            self.data = dict()

    # MAKE DATASET AND SAVE
    def make_dataset(self, coco_ann_path, coco_search_ann_path, images_path):
            image_names = list()

            # 1
            images_paths = get_files(images_path)
            for image in images_paths:
                image_names.append(image.split('/')[-1])

            coco_search_ann_paths = get_files(coco_search_ann_path)
            complete_fixation_data = []
            
            # 2
            for path in coco_search_ann_paths:
                with open(path) as f:
                    fixation_data = json.load(f)
                    complete_fixation_data += fixation_data
                
            # 3
            coco_ann_paths = get_files(coco_ann_path)
            for path in coco_ann_paths:
                # name of the annotation file
                ann_name = path.split('/')[-1] + '_annotations'
                
                # load the annotation file 
                with open(path) as f:
                    coco_ann = json.load(f)

                    # iterate over the images in the annotation file
                    for image in tqdm(coco_ann['images']):
                        image_id = image['id']
                        filename = image['file_name']
                        # check if the image is in the images folder
                        if filename in image_names:

                            if filename not in self.data.keys():
                                self.data[filename] = dict()
                                self.data[filename]['fixations'] = list()

                            if ann_name not in self.data[filename].keys():
                                self.data[filename][ann_name] = list()
                            
                            for fix in complete_fixation_data:
                                if fix["name"] == filename:
                                    self.data[filename]['fixations'].append(fix)

                            for ann in coco_ann['annotations']:
                                if ann['image_id'] == image_id:
                                    self.data[filename][ann_name].append(ann)

    def save_dataset(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=4)
    