# Define clip model
from transformers import CLIPVisionModel
import torch
from config import * 
import torch.nn.functional as F
class ClipModelWithClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(ClipModelWithClassifier, self).__init__()
        self.clip_model = CLIPVisionModel.from_pretrained(model_checkpoint, cache_dir=cache_dir)
        #for param in self.clip_model.parameters():
        #    param.requires_grad = False
        #self.fc1 = torch.nn.Linear(in_features=768, out_features=600)
        #self.fc2 = torch.nn.Linear(in_features=600, out_features=500)
        self.classifier_head = torch.nn.Linear(in_features=768, out_features=num_labels)

    def forward(self, input):
        outputs = self.clip_model(**input)
        #pooled_output = outputs.pooler_output  # pooled CLS states
        last_hidden_state = outputs.last_hidden_state
        pooled_last_hidden_state = last_hidden_state.mean(dim=1) 
        #x = F.relu(self.fc1(pooled_last_hidden_state))
        #x = F.relu(self.fc2(x))
        logits = self.classifier_head(pooled_last_hidden_state)
        probabilities = torch.softmax(logits, dim=-1)
        return logits
