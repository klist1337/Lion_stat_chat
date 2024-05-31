import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import string  # Importing string module
from model import NeuralNet
from nltk_utils import *

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


with open('assurance.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)


def clean_text(text):
    # Convertir en minuscules
    text = text.lower()
   
    # Supprimer les ponctuations
    text = ''.join([char for char in text if char not in string.punctuation])
   
    # Supprimer les caractères spéciaux
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
   
    # Supprimer les stopwords
    stop_words = set(stopwords.words('french'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
   
    return text

def clean_dataset(dataset):
    cleaned_dataset = {
        "intents": []
    }
   
    for intent in dataset["intents"]:
        cleaned_patterns = []
        for pattern in intent["patterns"]:
            cleaned_pattern = clean_text(pattern)
            cleaned_patterns.append(cleaned_pattern)
        intent["patterns"] = cleaned_patterns
        cleaned_dataset["intents"].append(intent)
   
    return cleaned_dataset


# Charger le fichier JSON initial
with open('assurance.json', 'r', encoding='utf-8') as file:
    your_dataset = json.load(file)
  

# Nettoyer le dataset
cleaned_dataset = clean_dataset(your_dataset)



# Enregistrer le dataset nettoyé dans un nouveau fichier JSON
with open('assurance_cleaned.json', 'w', encoding='utf-8') as output_file:
    json.dump(cleaned_dataset, output_file, indent=4, ensure_ascii=False)

print("Le fichier a été nettoyé et enregistré avec succès sous 'assurance_cleaned.json'.")

with open('assurance.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 32
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 32
output_size = len(tags)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.from_numpy(X_train).float()
        self.y_data = torch.from_numpy(y_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')