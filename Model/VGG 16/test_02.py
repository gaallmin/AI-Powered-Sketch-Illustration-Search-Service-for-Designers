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

# Use Gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VGG16((3, 224, 224), 20)

transformer = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

# Train loader
train_dataloader = Load(
        transformer,
        batch_size=128)
_, trainloader = train_dataloader("../train_val_test_dataset/train")

# Validation loader
valid_dataloader = Load(
        transformer,
        batch_size=128)
_, valloader = valid_dataloader("../train_val_test_dataset/valid")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=100,
        gamma=0.1)

# Init Trainer
trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=exp_lr_scheduler,
        device=device)

# Train
trainer.train(
        1000,
        trainloader,
        valloader)
