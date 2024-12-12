import torch.nn as nn  # Import PyTorch's neural network module

# Define the QLearning class, which inherits from PyTorch's nn.Module
class QLearning(nn.Module):
    def __init__(self):
        # Initialize the parent class
        super(QLearning, self).__init__()

        # First layer: Linear transformation from 4 inputs to 64 outputs, followed by a ReLU activation
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        
        # Second layer: Linear transformation from 64 inputs to 64 outputs, followed by a ReLU activation
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        
        # Third layer: Linear transformation from 64 inputs to 1 output
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        # Initialize weights for all layers
        self.initialize_weights()

    # Forward pass through the network
    def forward(self, input_data):
        # Pass the input through the first layer
        input_data = self.conv1(input_data)
        
        # Pass the result through the second layer
        input_data = self.conv2(input_data)
        
        # Pass the result through the third layer to produce the final output
        input_data = self.conv3(input_data)

        # Return the final output
        return input_data

    # Initialize the weights of the network using Xavier initialization
    def initialize_weights(self):
        for module in self.modules():  # Iterate through all modules in the model
            if isinstance(module, nn.Linear):  # Check if the module is a Linear layer
                # Apply Xavier uniform initialization to the layer's weights
                nn.init.xavier_uniform_(module.weight)
                # Initialize the layer's biases to zero
                nn.init.constant_(module.bias, 0)
