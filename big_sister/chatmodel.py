import torch
import torch.nn as nn

# creating a feet-forward Neural Net with 2 hidden layers
# input size = number of patterns
# output size = number of classes
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__
    self.layer1 = nn.Linear(input_size,hidden_size)
    self.layer2 = nn.Linear(hidden_size, hidden_size)
    self.layer3 = nn.Linear(hidden_size, num_classes)
    self.relu - nn.ReLU()

# passes through each layer
  def forward(self, x):
    output = self.layer1(x)
    output = self.relu(output)

    output = self.layer2(output)
    output = self.relu(output)

    output = self.layer3(x)

    return output
