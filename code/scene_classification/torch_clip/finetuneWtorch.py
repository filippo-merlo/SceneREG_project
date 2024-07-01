#%%
from nnet import *
from dataset import CollectionsDataset, final_dataset, processor
from dataset import *
from config import *
import sys
sys.path.append('./')
from dataset_prep import *

from torch.utils.data import DataLoader
# wandb
import wandb
wandb.login()
model_checkpoint = "openai/clip-vit-base-patch32"
config = {
    "model_checkpoint": model_checkpoint,
    "batch_size": 16,
    "num_epochs": 10,
    "lr": 1e-3,
    "momentum": 0.9
}

wandb.init(project=model_checkpoint.split('/')[1], config=config)

# Initialize DataLoader
# define preprocess
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(model_checkpoint, cache_dir=cache_dir)

train_dataloader = DataLoader(CollectionsDataset(final_dataset['train'], processor), shuffle=True, batch_size=wandb.config['batch_size'])
eval_dataloader = DataLoader(CollectionsDataset(final_dataset['test'], processor), shuffle=True, batch_size=wandb.config['batch_size'])

# Initialize model
n_labels = len(final_dataset['train'].features['scene_category'].names)
model = ClipModelWithClassifier(n_labels)

# # Create an optimizer and learning rate scheduler to fine-tune the model. Let's use the AdamW optimizer from PyTorch:
from torch.optim import SGD

optimizer = SGD(model.parameters(), lr=wandb.config['lr'], momentum=wandb.config['momentum'])

#Create the default learning rate scheduler from Trainer:
from transformers import get_scheduler
num_epochs = wandb.config['num_epochs']
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)

# specify device to use a GPU if you have access to one. Otherwise, training on a CPU may take several hours instead of a couple of minutes.
import torch

#if torch.backends.mps.is_available():
#   device = torch.device("mps")
#   print('mps Ok')
#else:
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Magic
log_freq = 100
wandb.watch(model, log_freq=log_freq)

# To keep track of your training progress, use the tqdm library to add a progress bar over the number of training steps:
from tqdm.auto import tqdm
from tqdm import tqdm

progress_bar = tqdm(range(num_training_steps))

import evaluate

metric_test = evaluate.load("accuracy", cache_dir=cache_dir)
metric_train = evaluate.load("accuracy", cache_dir=cache_dir)

criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):

    model.eval()
    for batch in tqdm(eval_dataloader):
        # move data to device
        labels = batch['labels'].to(device)
        input = {k:v.squeeze().to(device) for k, v in batch['image'].items()}
        with torch.no_grad():
            outputs = model(input)
        # eval
        predictions = torch.argmax(outputs, dim=-1)
        labels = torch.argmax(labels, dim= -1)
        metric_test.add_batch(predictions=predictions, references=labels)
    wandb.log({'eval_acc' : metric_test.compute()['accuracy']})

    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        # move data to device
        labels = batch['labels'].to(device)
        input = {k:v.squeeze().to(device) for k, v in batch['image'].items()}
        # forward pass
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        # eval
        labels = torch.argmax(labels, dim= -1)
        predictions = torch.argmax(outputs, dim=-1)
        metric_train.add_batch(predictions=predictions, references=labels)
        if batch_idx % log_freq == 0:
            wandb.log({"loss": loss})
            wandb.log({'train_acc' : metric_train.compute()['accuracy']})

   