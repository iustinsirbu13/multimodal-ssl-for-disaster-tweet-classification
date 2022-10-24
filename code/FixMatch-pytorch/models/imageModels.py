import torch
import torch.nn as nn
from torchvision import models


class ImageModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.model == 'resnet18':
            model = models.resnet18(pretrained=True)
            embedding_size = 512
        elif args.model == 'resnet152':
            model = models.resnet152(pretrained=True)
            embedding_size = 2048
        elif args.model == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=True)
            embedding_size = 1536
        else:
            raise ValueError('Unsupported model.')

        modules = list(model.children())[:-1]

        if args.dropout:
            modules.append(nn.Dropout(args.dropout))

        self.encoder = nn.Sequential(*modules)
        self.classifier = nn.Linear(embedding_size, args.num_classes)

    def forward(self, x): # (B x C x H x W)
        embedding = self.encoder(x) # (B x embedding_size)
        embedding = torch.flatten(embedding, start_dim=1)
        logits = self.classifier(embedding) # (B x num_classes,)
        return logits # B -> [-1.21, 2.123] Daca aplici Softmax se transforma in probabilitati
