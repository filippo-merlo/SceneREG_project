import os
path = '/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data/generated_images/'
sen_scene_to_keep = ['kitchen','living_room','home_office','bathroom','restaurant','bedroom','delicatessen','bakery/shop','pantry','coffee_shop','cubicle/office','banquet_hall','market/outdoor','dorm_room','dining_room','music_studio','art_studio','parking_lot','street','tower','fastfood_restaurant','train_railway','physics_laboratory','arrival_gate/outdoor','gas_station']

for sen_scene in sen_scene_to_keep:
    name_folder = sen_scene.replace('/','_')
    folder_path = os.path.join(path, name_folder)
    os.makedirs(folder_path, exist_ok=True)

    