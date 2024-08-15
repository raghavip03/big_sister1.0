import json
from utils import tokenize, bag_of_words, stem
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chatmodel import NeuralNet

with open('intents.json','r') as f:
  intents = json.load(f)

# store the words from the prompt
all_words = []
tags = []
patterns_tags = []

#-------------Tokenize-----------#
for intent in intents['intents']:
  #retrieve tag and append it to tags array
  tag = intent['tag']
  tags.append(tag)
  # retrieve each pattern, tokenize, and add it to all_words array
  # and append a (pattern, tag) tuple to the pattern_tag array
  for pattern in intent['patterns']:
    tokenized_pattern = tokenize(pattern)
    all_words.extend(tokenized_pattern)
    patterns_tags.append((tokenized_pattern, tag))

# exclude puntuation
ignore = ['?', '!', '.', ',']

#--------------Stemming and Sorting -----------------#
all_words = [stem(word) for word in all_words if word not in ignore]
all_words = sorted(set(all_words)) #remove duplicates

#--------------Training Data-------------------------#
pattern_train = []
tags_train = []

for (tokenized_pattern, tag) in patterns_tags:
  bag = bag_of_words(tokenized_pattern, all_words)
  pattern_train.append(bag)

  rep = tags.index(tag) # rep is 1 if tag exsists and 0 if it doesn't
  tags_train.append(rep)

pattern_train = np.array(pattern_train)
tags_train = np.array(tags_train)

batch_size = 8
# hidden_size = 8
# output_size = len(tags)
# input_size = len(all_words)
# print(input_size, len(all_words))
# print(output_size, tags)

#----------Creating a Dataset------------#
class ChatDataset(Dataset):
  def __init__(self):
    self.n_samples = len(tags_train)
    self.pattern_data = pattern_train
    self.tags_data = tags_train

    def __getdata__(self, index):
      return self.pattern_data[index], self.tags_data[index]

    def __len__(self):
      return self.n_samples

# training data


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# chatmodel = NeuralNet(input_size, hidden_size, output_size)s