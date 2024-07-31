#%%
import json
from pprint import pprint

#%%
with open('/Users/filippomerlo/Desktop/Datasets/sceneREG_data/visual_genome/objects.json', 'r') as f:
    objects = json.load(f)
#%%
pprint(objects)
#%%
def bbox2ratio(bbox):
    x,y,w,h = bbox
    width = w + x
    height = h + y
    return width/height

def add_and_avg(n, nn):
    return (n+nn)/2

#%%
object_avg_ratios = {}
for image in objects:
    for object in image['objects']:
        try:
            object_name = object['synsets'][0].split('.')[0].replace('_', ' ')
            if len(object['synsets']) > 1:
                continue
            if object_name not in object_avg_ratios.keys():
                object_avg_ratios[object_name] = 0
            object_bbox = [object['x'], object['y'], object['w'], object['h']]
            ratio = bbox2ratio(object_bbox)
            object_avg_ratios[object_name] = add_and_avg(object_avg_ratios[object_name], ratio)
        except:
            continue
#%%
pprint(object_avg_ratios)    

#%%
import pandas as pd
df = pd.read_csv('/Users/filippomerlo/Desktop/Datasets/sceneREG_data/THINGS/THINGSplus/Metadata/Concept-specific/size_meanRatings.tsv', sep='\t')
things_names = set(list(df['WordContext']))
#%%
len(things_names)
#%%
clean_things_names = []
for t in things_names:
    clean_things_names.append(t.split(' (')[0].replace(' ',''))
pprint(clean_things_names)
#%%
i = 0
for object in  object_avg_ratios.keys():
    if object.replace(' ','') in clean_things_names:
        i += 1
print(i)