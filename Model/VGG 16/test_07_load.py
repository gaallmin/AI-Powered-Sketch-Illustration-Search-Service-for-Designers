import sys
sys.path.append("../")

from train import Trainer
from model import VGG16
from load import Load

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

if __name__ == "__main__":

# Use Gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGG16((3, 224, 224), 20)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

# Init Trainer
    trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            )

    trainer.load("./saved_models/test_07.obj")

    trainer.loss_graph()
    trainer.f1_graph()
