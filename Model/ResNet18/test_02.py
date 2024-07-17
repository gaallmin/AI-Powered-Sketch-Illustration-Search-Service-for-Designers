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

# Use Gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet18((3, 224, 224), 20)

transformer = [
    transforms.Resize(256), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

# Train loader
train_dataloader = Load(
        transformer,
        num_workers=16,
        batch_size=256)
_, trainloader = train_dataloader("../random_augmented_dataset/train")

# Validation loader
valid_dataloader = Load(
        transformer,
        num_workers=16,
        batch_size=256)
_, valloader = valid_dataloader("../random_augmented_dataset/valid")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

# Init Trainer
trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        )

# Train
trainer.train(
        3000,
        trainloader,
        valloader,
        autosave_params={
            'use_autosave': True,
            'save_dir': './saved_models/test_02.obj',
            },
        )
