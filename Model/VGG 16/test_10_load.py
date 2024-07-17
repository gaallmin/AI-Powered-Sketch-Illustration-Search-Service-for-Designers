import sys
sys.path.append("../")

from train import Trainer
from model import VGG16
from load import Load
from utils import FocalLoss

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

if __name__ == "__main__":

    transformer = [
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

# Validation loader
    valid_dataloader = Load(
            transformer,
            num_workers=2,
            batch_size=32)
    _, valloader = valid_dataloader("../random_augmented_dataset_v3/valid")

# Use Gpu
    device = torch.device('cpu')
    model = VGG16((3, 224, 224), 20)

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-6)

# Init Trainer
    trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            )

# Train

    trainer.load("./saved_models/test_10.obj")

    '''
    trainer.loss_graph()
    trainer.f1_graph()

    trainer.f1_score(valloader)
    trainer.save_pth("./saved_models/test_10.pth")
    '''

    trainer.f1_score(valloader)
