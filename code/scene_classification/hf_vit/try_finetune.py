# WANDB
import wandb
wandb.login()
import os
project_name = 'vit_snacks_sweeps'
# Set a single environment variable
os.environ["WANDB_PROJECT"] = project_name
os.environ["WANDB_LOG_MODEL"] = 'true'
#%%
from transformers import ViTImageProcessor, ViTFeatureExtractor

cache_dir = '/mnt/cimec-storage6/users/filippo.merlo'
#cache_dir = '/Users/filippomerlo/Documents/GitHub/SceneReg_project/code/scene_classification/cache_dir'
#checkpoint = 'openai/clip-vit-large-patch14'
checkpoint = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(checkpoint, cache_dir= cache_dir)

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

datasets = load_dataset("scene_parse_150", cache_dir= cache_dir)
labels = datasets['train'].features['scene_category'].names

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x.convert('RGB') for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['scene_category']
    return inputs

datasets_processed = datasets.with_transform(transform)

import torch

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

# define function to compute metrics
import numpy as np
import evaluate

def compute_metrics(p):
    metric = evaluate.load("accuracy", cache_dir= cache_dir)
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def compute_metrics_fn(eval_preds):
  metrics = dict()
  
  accuracy_metric = evaluate.load('accuracy', cache_dir= cache_dir)
  precision_metric = evaluate.load('precision', cache_dir= cache_dir)
  recall_metric = evaluate.load('recall', cache_dir= cache_dir)
  f1_metric = evaluate.load('f1', cache_dir= cache_dir)


  logits = eval_preds.predictions
  labels = eval_preds.label_ids
  preds = np.argmax(logits, axis=-1)  
  
  metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
  metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
  metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
  metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))

  return metrics

# INIT MODEL
from transformers import ViTForImageClassification

id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

def model_init():
    vit_model = ViTForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir= cache_dir
    )
    return vit_model

## SWEEPS
# method
sweep_config = {
    'method': 'bayes',
    "metric": {"goal": "minimize", "name": "loss"},
}

# hyperparameters
parameters_dict = {
    'epochs': {
        'value': 10
        },
    'batch_size': {
        'values': [8, 16, 32, 64, 128, 256]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-3
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project=project_name)

from transformers import TrainingArguments, Trainer


def train(config=None):
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config

    # set training arguments
    training_args = TrainingArguments(
        output_dir=f'/mnt/cimec-storage6/users/filippo.merlo/{project_name}',
	    report_to='wandb',  # Turn on Weights & Biases logging
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        remove_unused_columns=False,
        fp16=True
    )

    # define training loop
    trainer = Trainer(
        # model,
        model_init=model_init,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=datasets_processed['train'],
        eval_dataset=datasets_processed['validation'],
        compute_metrics=compute_metrics_fn
    )


    # start training loop
    trainer.train()

wandb.agent(sweep_id, train, count=100)
