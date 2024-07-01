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

#pipeline =  AutoPipelineForInpainting.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16).to(device)
#pipeline.enable_model_cpu_offload()
## remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
#pipeline.enable_xformers_memory_efficient_attention()
#generator = torch.Generator("cuda").manual_seed(92)

# check predictions
def generate(init_image, target_box, new_object):
    # Given data
    x, y, w, h = target_box  # Coordinates and dimensions of the white box
    max_w, max_h = init_image.size  # Size of the image
    # Create a black background image
    mask = Image.new("RGB", (max_w, max_h), "black")
    # Create a drawing object
    draw = ImageDraw.Draw(mask)
    # Define the coordinates of the white box
    left = x
    top = y
    right = x + w
    bottom = y + h
    # Draw the white box on the black background
    draw.rectangle([left, top, right, bottom], outline="white", fill="white")
    # Save or display the image
    blurred_mask = pipeline.mask_processor.blur(mask, blur_factor=33)
    prompt = f"a {new_object}, realistic, highly detailed, 8k"
    negative_prompt = "bad anatomy, deformed, ugly, disfigured"
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=blurred_mask, generator=generator).images[0]
    grid_image = make_image_grid([init_image, blurred_mask, image], rows=1, cols=3)
    grid_image.show()

def make_image_grid(images, rows, cols):
        """
        Create a grid of images.
        
        :param images: List of PIL Image objects.
        :param rows: Number of rows in the grid.
        :param cols: Number of columns in the grid.
        :return: PIL Image object representing the grid.
        """
        assert len(images) == rows * cols, "Number of images does not match rows * cols"

        # Get the width and height of the images
        width, height = images[0].size
        
        # Create a new blank image with the correct size
        grid_width = width * cols
        grid_height = height * rows
        grid_image = Image.new('RGB', (grid_width, grid_height))

        # Paste the images into the grid
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid_image.paste(img, (col * width, row * height))

        return grid_image

class Dataset:

    def __init__(self, dataset_path = None):
        if dataset_path:
            with open(dataset_path) as f:
                self.data = json.load(f)
        else:
            self.data = dict()

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
    
    def visualize_img(self, img_name = None):
        
        if img_name != None:
            image = self.data[img_name]
        else:
            while img_name == None:
                img_name = rn.choice(list(self.data.keys()))
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
                if target == None:
                    img_name = None
                
        print('*',target)
        images_paths = get_files(images_path)
        image_picture = None
        for image_path in images_paths:
            if img_name in image_path:
                image_picture = Image.open(image_path)
                break
        # Convert PIL image to OpenCV format
        image_cv2 = cv2.cvtColor(np.array(image_picture), cv2.COLOR_RGB2BGR)

        # Draw the box of the image
        ann_key = 'instances_train2017_annotations'
        try:
            image[ann_key]
        except:
            ann_key = 'instances_val2017_annotations'

        target_bbox = None
        for ann in image[ann_key]:
            id = ann['category_id']
            color = (255, 0, 0)  # Red color
            object_names = list()
            for cat in coco_categories:
                if cat['id'] == id:
                    cat_name = cat['name']
                    object_names.append(cat_name)
            if target in object_names:
                color = (0, 0, 255)
                target_bbox = ann['bbox']
                target_segmentation = ann['segmentation']
                target_area = ann['area']
            x, y, width, height = ann['bbox']
            thickness = 2
            cv2.rectangle(image_cv2, (int(x), int(y)), (int(x + width), int(y + height)), color, thickness)

        # retrieve captions
        image_captions = []
        cap_key = 'captions_train2017_annotations'
        try:
            image[cap_key]
        except:
            cap_key = 'captions_val2017_annotations'
        for ann in image[cap_key]:
            caption = ann['caption']
            print(caption)
            image_captions.append(caption)

        # observe results
        # Convert back to PIL format for displaying
        image_with_box = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    
        # Display the image with the box
        plt.imshow(image_with_box)
        plt.axis('off')  # Turn off axis
        plt.show()

        # Crop
        # Segmentation
        image_mask_cv2 = cv2.cvtColor(np.array(image_picture), cv2.COLOR_RGB2BGR)
        target_segmentation = np.array(target_segmentation, dtype=np.int32).reshape((-1, 2))
        # Create a mask
        target_mask = np.zeros(image_mask_cv2.shape[:2], dtype=np.uint8)
        cv2.fillPoly(target_mask, [target_segmentation], 255)
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image_mask_cv2, image_mask_cv2, mask=target_mask)
        # Crop image 
        # Box
        x,y,w,h = target_bbox
        max_w, max_h = image_picture.size
        x_c = subtract_in_bounds(x,20)
        y_c = subtract_in_bounds(y,20)
        w_c = add_in_bounds(x,w+20,max_w)
        h_c = add_in_bounds(y,h+20,max_h)
        cropped_masked_image = masked_image[y_c:h_c, x_c:w_c]
        # Step 3: Convert the cropped image from BGR to RGB
        cropped_masked_image_rgb = cv2.cvtColor(cropped_masked_image, cv2.COLOR_BGR2RGB)
        # Step 4: Convert the cropped image to a PIL image
        cropped_masked_image_pil = Image.fromarray(cropped_masked_image_rgb)
        # Show
        plt.imshow(cropped_masked_image_pil)
        plt.axis('off')  # Turn off axis
        plt.show()

        cropped_image = image_picture.crop((x_c,y_c,w_c,h_c))
       
        # Classify scene
        #classify_scene_clip_llava(image_picture, scene_labels_context)
        scene_category = classify_scene_vit(image_picture)
        print(scene_category)
        # retrieve info from obscene
        objects_to_replace = find_object_to_replace(target, scene_category)
        print(objects_to_replace)
        images_paths = compare_imgs(cropped_masked_image_pil, objects_to_replace)
        #generate(image_picture, target_bbox, objects_to_replace[0])
        visualize_images(images_paths)

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
                    
                images_paths = get_files(images_path)
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

    
dataset = Dataset(dataset_path = '/Users/filippomerlo/Desktop/Datasets/sceneREG_data/coco_search18/coco_search18_annotated.json')

#%%
count, label_with_paths = dataset.get_scene_predictions()
pprint(count)
pprint(label_with_paths)

#%%
dataset.visualize_img()

#%%
print_dict_structure(dataset.data)
