# CONFIG FILE WITH ALL THE PATHS AND VARIABLES NEEDED
#%%
import pandas as pd
import pickle as pkl
import os
import json

# DECLARE PATHS
CACHE_DIR_SHARED = '/mnt/cimec-storage6/shared'
CACHE_DIR_PRIVATE = '/mnt/cimec-storage6/users/filippo.merlo'

# Path for the data folder
data_folder_path = '/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data/'

# Paths for making the dataset 
coco_ann_path = os.path.join(data_folder_path,'coco_search18/coco_annotations')
coco_search_ann_path = os.path.join(data_folder_path,'coco_search18/coco_search18_TP')
coco_images_path = os.path.join(data_folder_path,'coco_search18/images')

# Paths for the already made dataset 
dataset_path = os.path.join(data_folder_path,'coco_search18/coco_search18_annotated.json')

# Path for mappings
map_coco2things_path = os.path.join(data_folder_path,'mappings/map_coco2things.json')

# Path for object scene similarity metrics matices and norms
llama_ade_object_scene_similarities_path = os.path.join(data_folder_path,'scene_object_sim/llama3_8b_instruct_THINGSobject_scene_norms.pkl')
things_plus_size_mean_path = os.path.join(data_folder_path,'THINGSplus/Metadata/Concept-specific/size_meanRatings.tsv')

# Path for images
things_images_path = os.path.join(data_folder_path,'THINGS/Images')
# IMPORT DATA
# Object scene similarity metrics matices and norms 
things_plus_size_mean_matrix = pd.read_csv(things_plus_size_mean_path, sep='\t', engine='python', encoding='utf-8')
with open(llama_ade_object_scene_similarities_path, 'rb') as f:
    llama_norms = pkl.load(f)

# Mappings
with open(map_coco2things_path, 'r') as f:
    map_coco2things = json.load(f)

# NAMES AND CATEGORIES
# Scene Names
# categories of the SUN397 dataset
sun_scene_cat = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'apartment_building/outdoor', 'apse/indoor', 'aquarium', 'aqueduct', 'arch', 'archive', 'arrival_gate/outdoor', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'athletic_field/outdoor', 'atrium/public', 'attic', 'auditorium', 'auto_factory', 'badlands', 'badminton_court/indoor', 'baggage_claim', 'bakery/shop', 'balcony/exterior', 'balcony/interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'basketball_court/outdoor', 'bathroom', 'batters_box', 'bayou', 'bazaar/indoor', 'bazaar/outdoor', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'bistro/indoor', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth/indoor', 'botanical_garden', 'bow_window/indoor', 'bow_window/outdoor', 'bowling_alley', 'boxing_ring', 'brewery/indoor', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'cabin/outdoor', 'cafeteria', 'campsite', 'campus', 'canal/natural', 'canal/urban', 'candy_store', 'canyon', 'car_interior/backseat', 'car_interior/frontseat', 'carrousel', 'casino/indoor', 'castle', 'catacomb', 'cathedral/indoor', 'cathedral/outdoor', 'cavern/indoor', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'chicken_coop/indoor', 'chicken_coop/outdoor', 'childs_room', 'church/indoor', 'church/outdoor', 'classroom', 'clean_room', 'cliff', 'cloister/indoor', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'control_tower/outdoor', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'covered_bridge/exterior', 'creek', 'crevasse', 'crosswalk', 'cubicle/office', 'dam', 'delicatessen', 'dentists_office', 'desert/sand', 'desert/vegetation', 'diner/indoor', 'diner/outdoor', 'dinette/home', 'dinette/vehicle', 'dining_car', 'dining_room', 'discotheque', 'dock', 'doorway/outdoor', 'dorm_room', 'driveway', 'driving_range/outdoor', 'drugstore', 'electrical_substation', 'elevator/door', 'elevator/interior', 'elevator_shaft', 'engine_room', 'escalator/indoor', 'excavation', 'factory/indoor', 'fairway', 'fastfood_restaurant', 'field/cultivated', 'field/wild', 'fire_escape', 'fire_station', 'firing_range/indoor', 'fishpond', 'florist_shop/indoor', 'food_court', 'forest/broadleaf', 'forest/needleleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'garage/indoor', 'garbage_dump', 'gas_station', 'gazebo/exterior', 'general_store/indoor', 'general_store/outdoor', 'gift_shop', 'golf_course', 'greenhouse/indoor', 'greenhouse/outdoor', 'gymnasium/indoor', 'hangar/indoor', 'hangar/outdoor', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'hot_tub/outdoor', 'hotel/outdoor', 'hotel_room', 'house', 'hunting_lodge/outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'ice_skating_rink/indoor', 'ice_skating_rink/outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn/outdoor', 'islet', 'jacuzzi/indoor', 'jail/indoor', 'jail_cell', 'jewelry_shop', 'kasbah', 'kennel/indoor', 'kennel/outdoor', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'labyrinth/outdoor', 'lake/natural', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'library/indoor', 'library/outdoor', 'lido_deck/outdoor', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'market/indoor', 'market/outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'moat/water', 'monastery/outdoor', 'mosque/indoor', 'mosque/outdoor', 'motel', 'mountain', 'mountain_snowy', 'movie_theater/indoor', 'museum/indoor', 'music_store', 'music_studio', 'nuclear_power_plant/outdoor', 'nursery', 'oast_house', 'observatory/outdoor', 'ocean', 'office', 'office_building', 'oil_refinery/outdoor', 'oilrig', 'operating_room', 'orchard', 'outhouse/outdoor', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage/indoor', 'parking_garage/outdoor', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pilothouse/indoor', 'planetarium/outdoor', 'playground', 'playroom', 'plaza', 'podium/indoor', 'podium/outdoor', 'pond', 'poolroom/establishment', 'poolroom/home', 'power_plant/outdoor', 'promenade_deck', 'pub/indoor', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall/indoor', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'stadium/baseball', 'stadium/football', 'stage/indoor', 'staircase', 'street', 'subway_interior', 'subway_station/platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_pool/indoor', 'swimming_pool/outdoor', 'synagogue/indoor', 'synagogue/outdoor', 'television_studio', 'temple/east_asia', 'temple/south_asia', 'tennis_court/indoor', 'tennis_court/outdoor', 'tent/outdoor', 'theater/indoor_procenium', 'theater/indoor_seats', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'track/outdoor', 'train_railway', 'train_station/platform', 'tree_farm', 'tree_house', 'trench', 'underwater/coral_reef', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'volleyball_court/indoor', 'volleyball_court/outdoor', 'waiting_room', 'warehouse/indoor', 'water_tower', 'waterfall/block', 'waterfall/fan', 'waterfall/plunge', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'wine_cellar/barrel_storage', 'wine_cellar/bottle_storage', 'wrestling_ring/indoor', 'yard', 'youth_hostel']

# Object Names
# things dataset
things_plus_size_mean_matrix = pd.read_csv(things_plus_size_mean_path, sep='\t', engine='python', encoding='utf-8')
things_words_id = list(things_plus_size_mean_matrix['uniqueID'])
things_words_context = list(things_plus_size_mean_matrix['WordContext'])
things_object_cat = list(things_plus_size_mean_matrix['Word'])

# coco dataset
coco_object_cat =  [{"supercategory": "person","id": 1,"name": "person"},{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]

# INIT MODELS
# set devices
device = 'cuda'

# Initialize the model for scene categorization
import wandb
import torch
from transformers import AutoImageProcessor, ViTForImageClassification

# Create the label to ID mapping
label2id = {label: idx for idx, label in enumerate(sun_scene_cat)}
# Reverse the mapping to create ID to label mapping
id2label = {idx: label for label, idx in label2id.items()}

with wandb.init(project="vit-base-patch16-224_SUN397") as run:
    # Pass the name and version of Artifact
    my_model_name = "model-on5m6wmj:v0"
    my_model_artifact = run.use_artifact(my_model_name)

    # Download model weights to a folder and return the path
    model_dir = my_model_artifact.download(CACHE_DIR_PRIVATE)

    # Load your Hugging Face model from that folder
    #  using the same model class
    vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", cache_dir=CACHE_DIR_PRIVATE)
    vit_model = ViTForImageClassification.from_pretrained(
        model_dir,
        num_labels=len(sun_scene_cat),
        id2label=id2label,
        label2id=label2id,
        cache_dir=CACHE_DIR_PRIVATE
    ).to(device)

# Initialize model for image embedding
from transformers import AutoImageProcessor, ViTModel 
vitc_image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=CACHE_DIR_PRIVATE)
vitc_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=CACHE_DIR_SHARED).to(device)

# Initialize model for generation
from diffusers import AutoPipelineForInpainting
pipeline =  AutoPipelineForInpainting.from_pretrained("JunhaoZhuang/PowerPaint-v2-1", cache_dir = CACHE_DIR_SHARED,  torch_dtype=torch.float16).to(device)
pipeline.enable_model_cpu_offload()

# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
generator = torch.Generator("cuda").manual_seed(92)

