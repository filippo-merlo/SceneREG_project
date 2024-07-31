#%% Import dataset
import pandas as pd
file_path = '/Users/filippomerlo/Desktop/Datasets/ObScene/ObSceneDatabase_SupplementalTables.xlsx'
excel_data = pd.read_excel(file_path, sheet_name='Table S5 - RELATIONSHIPS(898)')
# congruency in a scale of 0-5
#%% Select columns of interest: 'Relationship Type', 'Object', 'Object Number', 'Scene', 'Scene Number', 'Congruency_Mean'
data = excel_data[['Relationship Type', 'Object', 'Object Number', 'Scene', 'Scene Number', 'Congruency_Mean']]
data
#%% Get Scene and Object names
scenes = list(set([x.replace('_B.jpg', '').replace('_A.jpg', '') for x in data['Scene'].unique()]))
objects = [x.replace('.jpg','') for x in data['Object'].unique()]
scenes_n = [x for x in range(len(scenes))]
objects_n = [x for x in range(len(objects))]
#%%
objects
