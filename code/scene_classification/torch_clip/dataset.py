import torch
from torch.utils.data import Dataset

# Split the dataset
### Define Collection Dataset
import torch
from torch.utils.data import Dataset

class CollectionsDataset(Dataset):
    def __init__(self, 
                 hf_dataset, 
                 transform=None):
        self.data = hf_dataset
        self.transform = transform
        self.num_classes = len(self.data.features['scene_category'].names)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['scene_category']
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1

        if self.transform:
            image = self.transform(images=image, return_tensors='pt')

        return {'image': image,
                'labels': label_tensor
                }




