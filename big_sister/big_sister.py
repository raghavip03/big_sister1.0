
# implementing the chat model
import random
import json
import torch
from chatmodel import NeuralNet
from utils import tokenize, bag_of_words

dev = torch.device('cuba' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
  intents = json.load(f)

# loading trained data in
FILE = "model_data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
chatmodel_state = data["chatmodel_state"]

chatmodel = NeuralNet(input_size, hidden_size, output_size).to(dev)
chatmodel.load_state_dict(chatmodel_state)
chatmodel.eval()

chatbot_name = "Big Sister"
print("Let's chat! type 'quit' to exit")
while True:
  sentence = input('You: ')
  if sentence == "quit":
    break

  sentence = tokenize(sentence)
  bag = bag_of_words(sentence, all_words)
  bag = bag.reshape(1, bag.shape[0])
  bag = torch.from_numpy(bag).to(dev) #convert to torch

  output = chatmodel(bag)
  _, predicted = torch.max(output, dim=1)
  tag = tags[predicted.item()]

  probs = torch.softmax(output, dim=1)
  probability = probs[0][predicted.item()]

  if probability.item() > 0.75:
    for intent in intents["intents"]:
      if tag == intent["tag"]:
        print(f"{chatbot_name}: {random.choice(intent["responses"])}")
  else:
    print(f"{chatbot_name}: I do not understand...")
