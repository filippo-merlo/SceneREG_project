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

import pandas as pd
# Load the size_mean_matrix file
size_mean_path = './things_tsv/size_meanRatings.tsv'
size_mean_matrix = pd.read_csv(size_mean_path, sep='\t', engine='python', encoding='utf-8')
things_words_id = list(size_mean_matrix['uniqueID'])
things_words_context = list(size_mean_matrix['WordContext'])
candidates = things_words_context

# Load scene categories from ADE20K hf
from datasets import load_dataset
ade_hf_data = load_dataset("scene_parse_150", cache_dir='/mnt/cimec-storage6/shared/hf_datasets')
scenes_categories = ade_hf_data['train'].features['scene_category'].names

sun_scene_cat = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'apartment_building/outdoor', 'apse/indoor', 'aquarium', 'aqueduct', 'arch', 'archive', 'arrival_gate/outdoor', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'athletic_field/outdoor', 'atrium/public', 'attic', 'auditorium', 'auto_factory', 'badlands', 'badminton_court/indoor', 'baggage_claim', 'bakery/shop', 'balcony/exterior', 'balcony/interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'basketball_court/outdoor', 'bathroom', 'batters_box', 'bayou', 'bazaar/indoor', 'bazaar/outdoor', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'bistro/indoor', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth/indoor', 'botanical_garden', 'bow_window/indoor', 'bow_window/outdoor', 'bowling_alley', 'boxing_ring', 'brewery/indoor', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'cabin/outdoor', 'cafeteria', 'campsite', 'campus', 'canal/natural', 'canal/urban', 'candy_store', 'canyon', 'car_interior/backseat', 'car_interior/frontseat', 'carrousel', 'casino/indoor', 'castle', 'catacomb', 'cathedral/indoor', 'cathedral/outdoor', 'cavern/indoor', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'chicken_coop/indoor', 'chicken_coop/outdoor', 'childs_room', 'church/indoor', 'church/outdoor', 'classroom', 'clean_room', 'cliff', 'cloister/indoor', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'control_tower/outdoor', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'covered_bridge/exterior', 'creek', 'crevasse', 'crosswalk', 'cubicle/office', 'dam', 'delicatessen', 'dentists_office', 'desert/sand', 'desert/vegetation', 'diner/indoor', 'diner/outdoor', 'dinette/home', 'dinette/vehicle', 'dining_car', 'dining_room', 'discotheque', 'dock', 'doorway/outdoor', 'dorm_room', 'driveway', 'driving_range/outdoor', 'drugstore', 'electrical_substation', 'elevator/door', 'elevator/interior', 'elevator_shaft', 'engine_room', 'escalator/indoor', 'excavation', 'factory/indoor', 'fairway', 'fastfood_restaurant', 'field/cultivated', 'field/wild', 'fire_escape', 'fire_station', 'firing_range/indoor', 'fishpond', 'florist_shop/indoor', 'food_court', 'forest/broadleaf', 'forest/needleleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'garage/indoor', 'garbage_dump', 'gas_station', 'gazebo/exterior', 'general_store/indoor', 'general_store/outdoor', 'gift_shop', 'golf_course', 'greenhouse/indoor', 'greenhouse/outdoor', 'gymnasium/indoor', 'hangar/indoor', 'hangar/outdoor', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'hot_tub/outdoor', 'hotel/outdoor', 'hotel_room', 'house', 'hunting_lodge/outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'ice_skating_rink/indoor', 'ice_skating_rink/outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn/outdoor', 'islet', 'jacuzzi/indoor', 'jail/indoor', 'jail_cell', 'jewelry_shop', 'kasbah', 'kennel/indoor', 'kennel/outdoor', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'labyrinth/outdoor', 'lake/natural', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'library/indoor', 'library/outdoor', 'lido_deck/outdoor', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'market/indoor', 'market/outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'moat/water', 'monastery/outdoor', 'mosque/indoor', 'mosque/outdoor', 'motel', 'mountain', 'mountain_snowy', 'movie_theater/indoor', 'museum/indoor', 'music_store', 'music_studio', 'nuclear_power_plant/outdoor', 'nursery', 'oast_house', 'observatory/outdoor', 'ocean', 'office', 'office_building', 'oil_refinery/outdoor', 'oilrig', 'operating_room', 'orchard', 'outhouse/outdoor', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage/indoor', 'parking_garage/outdoor', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pilothouse/indoor', 'planetarium/outdoor', 'playground', 'playroom', 'plaza', 'podium/indoor', 'podium/outdoor', 'pond', 'poolroom/establishment', 'poolroom/home', 'power_plant/outdoor', 'promenade_deck', 'pub/indoor', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall/indoor', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'stadium/baseball', 'stadium/football', 'stage/indoor', 'staircase', 'street', 'subway_interior', 'subway_station/platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_pool/indoor', 'swimming_pool/outdoor', 'synagogue/indoor', 'synagogue/outdoor', 'television_studio', 'temple/east_asia', 'temple/south_asia', 'tennis_court/indoor', 'tennis_court/outdoor', 'tent/outdoor', 'theater/indoor_procenium', 'theater/indoor_seats', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'track/outdoor', 'train_railway', 'train_station/platform', 'tree_farm', 'tree_house', 'trench', 'underwater/coral_reef', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'volleyball_court/indoor', 'volleyball_court/outdoor', 'waiting_room', 'warehouse/indoor', 'water_tower', 'waterfall/block', 'waterfall/fan', 'waterfall/plunge', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'wine_cellar/barrel_storage', 'wine_cellar/bottle_storage', 'wrestling_ring/indoor', 'yard', 'youth_hostel']
scenes_categories = sun_scene_cat

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
            prompt = "Is it possible to find the object '{word}' in the place '{scene}'?".format(word=single_candidate, scene=scene_name.replace("_", " ").replace("/", " "))
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

with open('/mnt/cimec-storage6/users/filippo.merlo/llama3_8b_instruct_THINGSobject_scene_norms.pkl', 'wb') as f:
    pkl.dump(answers, f)
