
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data

from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, FlaubertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import pandas as pd
import os
import numpy as np
import json
from torchvision import transforms
from accelerate import DataLoaderConfiguration
from accelerate import Accelerator
import evaluate

##INITIALISATION DES DONNEES #

# IMPORTER LES DATASETS # 
# Donnees Train # 
data_files_train = {"train": "data_train_mini.jsonl"} 
train_dataset = load_dataset('json', data_files=data_files_train) 

# Donnees Test # 
data_files_test = {"test": "data_test.jsonl"}
test_dataset = load_dataset('json', data_files=data_files_test)
# ajuster le type des colonnes du test dataset 
test_dataset.rename_column("label_id","label")

# INITIALISER LE TRAIN DATASET
label_list = sorted(train_dataset['train'].unique('label')) # extraire tous les categories de labels 
label_dict = {label: idx for idx, label in enumerate(label_list)}  #marquer les indices des categories de label 

# Modifier le dataset en replacant label strings des indices correspondant 
def label_2_id(example):
    """
    cette fonction sert a ajouter une colonne "label_id" qui stoke les indices des categories des labels 
    
    Args:
        example (str): rang de datasets

    Returns:
        example (str): rang de datasets
    """
    encoded_label = label_dict[example['label']]
    example['label_id'] = encoded_label
    return example

#initialiser les datasets en ajoutant label_id
train_dataset = train_dataset.map(label_2_id) 
 
# preparer les parametres qui doivent etre envoyes dans le module Flaubert 
num_labels = len(label_dict)
label2id = label_dict
id2label = {id: tag for tag, id in label2id.items()}


# INITIALISER LE TEST DATASET 


def add_label_id(example):
    """
    cette fonction vise a ajouter une nouvelle colonne qui stoke l'indice initiale des categories des labels 

    Args:
        example (str): rang de datasets

    Returns:
        example (str): rang de datasets
    """
    encode_label = 0
    example['label_id'] = encode_label 
    return example  

#initialiser les datasets en ajoutant label_id
test_dataset = test_dataset.map(add_label_id)


## PRETAITEMENT DES DONNEES #

# importer les tokenizer et le modele
tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")

def tokenize_function(example):
    """
    cette fonction sert a tokenizer les textes et remettre les lables et les vecteurs 

    Args:
        example(str): rang de datasets

    Returns:
        example(str): rangs ajoutes de datasets
    """
    tokenized_inputs = tokenizer(example['texte'], truncation=True, padding=True,max_length=128) #faire en sorte que les séquences soient de la même longueur
    example['labels'] = example['label_id'] 
    return {**tokenized_inputs, 'labels': example['labels']}


tokenized_datasets = train_dataset.map(tokenize_function, batched=True) # tokenizer et encoder les labels 
tokenized_datasets = tokenized_datasets.remove_columns(['texte', 'label','label_id', 'url', 'id']) # enlever les colonnes inutiles 
tokenized_datasets.set_format('torch', columns=['input_ids','attention_mask', 'token_type_ids','labels']) #recharger les colonnes necessaires comme les parametres envoyes dans le module 

tokenized_datasets_test = test_dataset.map(tokenize_function, batched=True) # tokenizer et encoder les labels 
tokenized_datasets_test = tokenized_datasets_test.remove_columns(['texte', 'label_id', 'url', 'id']) # enlever les colonnes inutiles 
tokenized_datasets.set_format('torch', columns=[ 'input_ids', 'attention_mask', 'token_type_ids','labels'])#recharger les colonnes necessaires comme les parametres envoyes dans le module 

# creer un bach d'exemples comme un des parametres envoyes dans le module 
data_collator =DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# importer un accelerateur pour executer plus rapidement les donnees 
dataLoader_conf = DataLoaderConfiguration(
    dispatch_batches=None,
    split_batches=False,
    even_batches=True,
    use_seedable_sampler=True
)

accelerator = Accelerator(dataloader_config= dataLoader_conf)

#importer les calcutes de la matrice 
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    """
    cette fonction vise a calculer la matrice de l'evaluation du module 
    
    Args:
        eval_pred (int, list): logits du module entraine et la liste de labels 

    Returns:
        "accuracy", "f1", "precision", "recall": les resultats de matrice 
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    return clf_metrics.compute(predictions=predictions, references=labels)



## ENTRAINEMENT DU MODULE # 
model = FlaubertForSequenceClassification.from_pretrained(
    "flaubert/flaubert_base_cased", num_labels=8, id2label=id2label, label2id=label2id

)

training_args = TrainingArguments(
                  output_dir='/content/results',
                  learning_rate=2e-5,
                  per_device_train_batch_size=8,
                  per_device_eval_batch_size=8,
                  num_train_epochs=2,
                  weight_decay=0.01,
                  evaluation_strategy="epoch",
                  save_strategy="epoch",
                  load_best_model_at_end=True,
                  )

trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"], #une classe de dataset
            eval_dataset=tokenized_datasets_test["test"], 
            tokenizer=tokenizer,
            data_collator = data_collator,
            compute_metrics=compute_metrics,
            )

trainer = accelerator.prepare(trainer) #accelerer l'entrainement 
trainer.train()
