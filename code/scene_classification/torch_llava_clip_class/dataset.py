# Split the dataset
### Define Collection Dataset
import torch
from torch.utils.data import Dataset

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class CollectionsDataset(Dataset):
    def __init__(self, 
                 hf_dataset, 
                 processor=None):
        
        self.data = hf_dataset
        if type(processor) == list:
            self.rep_list = processor
        else:
            self.clip = processor['clip_model']
            self.clip_processor = processor['clip_processor']
            self.llava_processor = processor['llava_processor']
            self.llava = processor['llava_model']

        self.prompt = "USER: <image>\nWhere is the picture taken?\nASSISTANT:"
        self.num_classes = len(self.data.features['scene_category'].names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['scene_category']
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1

        if self.rep_list:
            return {'reppresentation': self.rep_list[idx],
                    'labels': label_tensor}
        else:
            # process image and text
            with torch.no_grad():
                llava_inputs = self.llava_processor(self.prompt, image, return_tensors='pt').to(device0, torch.float16)
                llava_encode = self.llava.generate(**llava_inputs, max_new_tokens=200, do_sample=False)
                llava_caption = self.llava_processor.decode(llava_encode[0][2:], skip_special_tokens=True)
                inputs = self.clip_processor(text=str(llava_caption), images=image, return_tensors="pt", padding=True).to(device0)
                outputs = self.clip(**inputs)
                txt_features = outputs.text_model_output.last_hidden_state.mean(dim=1) 
                img_features = outputs.vision_model_output.last_hidden_state.mean(dim=1) 
                reppresentation = torch.cat([txt_features, img_features], dim=1).squeeze()

            return {'reppresentation': reppresentation,
                    'labels': label_tensor}




