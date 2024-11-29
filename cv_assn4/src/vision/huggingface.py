import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification, ConvNextForImageClassification

class MyHFModel(nn.Module):
    def __init__(self):
        """Initialize network layers using MobileViT from Hugging Face.

        Note: Freeze the layers except the last one.
        """
        super().__init__()
        self.hf_model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224")
        # self.hf_model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
        # self.hf_model = nn.Sequential(*list(mobilevit.children())[:-2])

        for param in self.hf_model.parameters():
            param.requires_grad = False
        print(self.hf_model)
        self.hf_model.classifier = nn.Linear(1024, 15)
        # self.hf_model.classifier = nn.Linear(640, 15)

        for param in self.hf_model.classifier.parameters():
            param.requires_grad = True

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
        """
        x = x.repeat(1, 3, 1, 1)
        outputs = self.hf_model(x)
        model_output = outputs.logits

        return model_output
