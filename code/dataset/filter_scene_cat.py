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

#%%
Counter({'kitchen': 598,
         'living_room': 486,
         'home_office': 425,
         'bathroom': 307,
         'restaurant': 232,
         'bedroom': 171,
         'delicatessen': 166,
         'bakery/shop': 121,
         'pantry': 86,
         'coffee_shop': 79,
         'cubicle/office': 78,
         'banquet_hall': 70,
         'restaurant_kitchen': 66,
         'market/outdoor': 58,
         'dorm_room': 53,
         'dining_room': 53,
         'music_studio': 53,
         'hotel_room': 42,
         'art_studio': 42,
         'kitchenette': 40,
         'parking_lot': 39,
         'street': 38,
         'cheese_factory': 36,
         'tower': 36,
         'crosswalk': 35,
         'fastfood_restaurant': 33,
         'galley': 29,
         'train_railway': 25,
         'computer_room': 25,
         'toyshop': 22,
         'gas_station': 21,
         'physics_laboratory': 20,
         'arrival_gate/outdoor': 20,
         'shopfront': 18,
         'office': 17,
         'highway': 17,
         'restaurant_patio': 17,
         'runway': 16,
         'parlor': 15,
         'florist_shop/indoor': 14,
         'vegetable_garden': 14,
         'waiting_room': 14,
         'diner/indoor': 12,
         'booth/indoor': 12,
         'jewelry_shop': 11,
         'butchers_shop': 11,
         'candy_store': 11,
         'operating_room': 10,
         'dining_car': 10,
         'bow_window/indoor': 10,
         'bar': 9,
         'airport_terminal': 9,
         'art_gallery': 9,
         'wet_bar': 8,
         'skyscraper': 8,
         'harbor': 8,
         'conference_room': 8,
         'corn_field': 8,
         'phone_booth': 8,
         'museum/indoor': 7,
         'residential_neighborhood': 7,
         'construction_site': 7,
         'dinette/home': 7,
         'wine_cellar/bottle_storage': 7,
         'youth_hostel': 7,
         'jail_cell': 6,
         'closet': 6,
         'water_tower': 6,
         'bridge': 6,
         'berth': 6,
         'beauty_salon': 6,
         'utility_room': 6,
         'amusement_park': 6,
         'alley': 6,
         'patio': 6,
         'supermarket': 6,
         'dinette/vehicle': 6,
         'doorway/outdoor': 6,
         'biology_laboratory': 6,
         'locker_room': 5,
         'playroom': 5,
         'motel': 5,
         'ice_cream_parlor': 5,
         'hospital_room': 5,
         'garage/indoor': 5,
         'vineyard': 5,
         'archive': 4,
         'bus_interior': 4,
         'apartment_building/outdoor': 4,
         'server_room': 4,
         'balcony/interior': 4,
         'mountain': 4,
         'ticket_booth': 4,
         'escalator/indoor': 4,
         'airplane_cabin': 4,
         'fire_station': 4,
         'brewery/indoor': 4,
         'orchard': 4,
         'plaza': 3,
         'hot_tub/outdoor': 3,
         'shed': 3,
         'childs_room': 3,
         'canal/urban': 3,
         'botanical_garden': 3,
         'general_store/indoor': 3,
         'bookstore': 3,
         'attic': 3,
         'wheat_field': 3,
         'chicken_coop/indoor': 3,
         'nursery': 3,
         'library/indoor': 3,
         'house': 3,
         'staircase': 3,
         'cemetery': 3,
         'laundromat': 3,
         'basement': 3,
         'ball_pit': 3,
         'arch': 2,
         'mosque/outdoor': 2,
         'picnic_area': 2,
         'herb_garden': 2,
         'office_building': 2,
         'subway_station/platform': 2,
         'forest_road': 2,
         'clothing_store': 2,
         'van_interior': 2,
         'sushi_bar': 2,
         'forest_path': 2,
         'pulpit': 2,
         'car_interior/frontseat': 2,
         'ocean': 2,
         'reception': 2,
         'car_interior/backseat': 2,
         'bazaar/outdoor': 2,
         'heliport': 2,
         'dentists_office': 2,
         'shower': 2,
         'subway_interior': 2,
         'fountain': 2,
         'labyrinth/outdoor': 2,
         'movie_theater/indoor': 2,
         'driving_range/outdoor': 2,
         'ski_slope': 2,
         'raceway': 2,
         'veterinarians_office': 2,
         'control_tower/outdoor': 2,
         'art_school': 2,
         'lobby': 2,
         'stage/indoor': 2,
         'train_station/platform': 2,
         'lake/natural': 2,
         'beach': 2,
         'bowling_alley': 2,
         'videostore': 2,
         'classroom': 2,
         'control_room': 2,
         'racecourse': 1,
         'fire_escape': 1,
         'ruin': 1,
         'track/outdoor': 1,
         'elevator_shaft': 1,
         'hotel/outdoor': 1,
         'lock_chamber': 1,
         'driveway': 1,
         'burial_chamber': 1,
         'cliff': 1,
         'abbey': 1,
         'clean_room': 1,
         'lido_deck/outdoor': 1,
         'kennel/outdoor': 1,
         'baggage_claim': 1,
         'game_room': 1,
         'synagogue/outdoor': 1,
         'thriftshop': 1,
         'general_store/outdoor': 1,
         'cottage_garden': 1,
         'theater/indoor_seats': 1,
         'aquarium': 1,
         'landfill': 1,
         'firing_range/indoor': 1,
         'chemistry_lab': 1,
         'building_facade': 1,
         'pub/indoor': 1,
         'balcony/exterior': 1,
         'golf_course': 1,
         'chicken_coop/outdoor': 1,
         'diner/outdoor': 1,
         'assembly_line': 1,
         'excavation': 1,
         'cockpit': 1,
         'hangar/indoor': 1,
         'medina': 1,
         'cavern/indoor': 1,
         'desert/sand': 1,
         'boat_deck': 1,
         'sandbar': 1,
         'volleyball_court/outdoor': 1,
         'jacuzzi/indoor': 1,
         'canal/natural': 1,
         'lighthouse': 1,
         'discotheque': 1,
         'landing_deck': 1,
         'field/wild': 1,
         'yard': 1,
         'television_studio': 1,
         'elevator/interior': 1,
         'parking_garage/indoor': 1,
         'pasture': 1,
         'baseball_field': 1,
         'music_store': 1,
         'snowfield': 1,
         'park': 1,
         'crevasse': 1,
         'playground': 1,
         'bayou': 1,
         'campsite': 1,
         'garbage_dump': 1,
         'market/indoor': 1})

#%% selected
selected = ['kitchen','living_room','home_office','bathroom','restaurant','bedroom','delicatessen','bakery/shop','pantry','coffee_shop','cubicle/office','banquet_hall','market/outdoor','dorm_room','dining_room','music_studio','art_studio','parking_lot','street','tower','fastfood_restaurant','train_railway','physics_laboratory','arrival_gate/outdoor','gas_station']




