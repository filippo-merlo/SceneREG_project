#%%
from config import *
from utils import *
from pprint import pprint
import json
from tqdm import tqdm
import random as rn
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter
from diffusers import AutoPipelineForInpainting

class Dataset:

    def __init__(self, dataset_path = None):
        if dataset_path:
            with open(dataset_path) as f:
                self.data = json.load(f)
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
    
   
dataset = Dataset(dataset_path = '/Users/filippomerlo/Desktop/Datasets/sceneREG_data/coco_search18/coco_search18_annotated.json')
print_dict_structure(dataset)