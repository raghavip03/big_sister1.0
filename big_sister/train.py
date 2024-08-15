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


# Hyperparameters set in place before training a model
# helps with controlling the learning process
batch_size = 8
hidden_size = 8
input_size = len(all_words) #input should be all words
output_size = len(tags)  #output should be the tags which these words are contained in
learning_rate = 0.001
num_epochs = 1000

#----------Creating the Dataset------------#
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(pattern_train)
        self.x_data = pattern_train
        self.y_data = tags_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# training data
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
chatmodel = NeuralNet(input_size, hidden_size, output_size).to(dev)

# loss and optimizer functions
# loss function measures our model's prediction with our targert
# optimizer function determines how our network will need to be updated based on the loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(chatmodel.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  for (words, reps) in train_loader:
    words = words.to(dev)
    reps = reps.to(dtype=torch.long).to(dev)

    # forward pass
    outputs = chatmodel(words)
    loss = criterion(outputs, reps)

    # optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss, loss=[{loss.item():.4f}]')

#------------------Save the data--------------------------#
data = {
   "chatmodel_state": chatmodel.state_dict(),
   "input_size": input_size,
   "output_size": output_size,
   "hidden_size": hidden_size,
   "all_words": all_words,
   "tags": tags
}

FILE = "model_data.pth"
torch.save(data, FILE)


