import sys
sys.path.append("../")

from train import Trainer
from model import VGG16_v2
from load import Load
from utils import FocalLoss

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

# Use Gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VGG16_v2((3, 224, 224), 20)

transformer = [
    transforms.Resize(256), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

# Train loader
train_dataloader = Load(
        transformer,
        num_workers=20,
        batch_size=128)
_, trainloader = train_dataloader("../random_augmented_dataset_v5/train")

# Validation loader
valid_dataloader = Load(
        transformer,
        num_workers=20,
        batch_size=128)
_, valloader = valid_dataloader("../random_augmented_dataset_v5/valid")

criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-7)

# Init Trainer
trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        )

trainer.load("./saved_models/test_02.obj")

# Train
trainer.train(
        60,
        trainloader,
        valloader,
        autosave_params={
            'use_autosave': True,
            'save_dir': './saved_models/test_02.obj',
            },
        )
