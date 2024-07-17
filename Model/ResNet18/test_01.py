import sys
sys.path.append("../")

from train import Trainer
from model import ResNet18
from load import Load

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms


# windows multiprocessing needs below if line
if __name__ == '__main__':

# Use Gpu
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = ResNet18((3, 224, 224), 20)

    transformer = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

# Train loader
    train_dataloader = Load(
            transformer,
            num_workers=2,
            batch_size=256)
    _, trainloader = train_dataloader("../augmented_dataset/train")

# Validation loader
    valid_dataloader = Load(
            transformer,
            num_workers=2,
            batch_size=256)
    _, valloader = valid_dataloader("../augmented_dataset/valid")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1)

    exp_lr_scheduler.last_epoch = 30 * 2 # only 4 times update

# Init Trainer
    trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=exp_lr_scheduler,
            device=device,
            )

# Train
    trainer.train(
            3000,
            trainloader,
            valloader,
            autosave_params={
                'use_autosave': True,
                'save_dir': './saved_models/test_01.obj',
                },
            )
