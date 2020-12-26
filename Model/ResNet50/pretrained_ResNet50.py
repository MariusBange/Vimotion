import torchvision.models as models
from torch import nn

class preResNet50(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-50 and frozen weights from the first layer to conv4 layer."""
        super(preResNet50, self).__init__()
        pretrained_model = models.resnet50(pretrained=True)
        for param in pretrained_model.parameters():  # freeze all parameters
            param.requires_grad = False

        modules = list(pretrained_model.children())[:-1]  # remove the last FC layer (classification model)

        self.modified_pretrained = nn.Sequential(*modules)


    def forward(self, images):
        """Extract feature vectors from input images."""
        ftrs = self.modified_pretrained(images)
        ftrs = ftrs.reshape(ftrs.size(0), -1)
        return ftrs

