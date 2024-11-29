import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super().__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.conv_layers = nn.Sequential(
            # Conv layer 1 with a larger filter (5x5)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),  
            nn.BatchNorm2d(16),  # BatchNorm after Conv layer 1
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=3),  # MaxPool (reduces to 32x32)
            nn.Dropout(p=0.3),  # Dropout for regularization
            
            # Conv layer 2 with a larger filter (5x5)
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5),  
            nn.BatchNorm2d(64),  # BatchNorm after Conv layer 2
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=3, stride=3),  # MaxPool (reduces to 16x16)
            nn.Dropout(p=0.3),  # Dropout for regularization
            
            # Conv layer 3 with a smaller filter (3x3)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  # BatchNorm after Conv layer 3
            nn.ReLU(),  # ReLU activation
            nn.Dropout(p=0.3)  # Final Dropout
        )

        # Define fully connected layers with Dropout
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=800, out_features=100),  # Linear layer 1 (adjusted for 8x8 spatial size)
            nn.ReLU(),  # ReLU activation
            nn.Linear(in_features=100, out_features=15)  # Linear layer 2 (final output layer)
        )

        # Loss criterion with mean reduction
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        
        x = self.conv_layers(x).reshape(x.shape[0], -1)
        model_output = self.fc_layers(x)
        
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
