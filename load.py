from typing import List

import torch
from torch.utils.data import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import ImageFile


class Load():

    def __init__(
            self,
            transformer=None,
            num_workers: int = 10,
            batch_size: int = 4,
            flatten: bool = False,
            ):

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if not transformer:
            transformer = [
                    transforms.ToTensor()
                    ]

        if flatten:
            transformer.append(transforms.Lambda(torch.flatten))

        self.transform = transforms.Compose(transformer)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(
            self,
            directory: str,
            balanced_sampling: bool = False):

        dataset = torchvision.datasets.ImageFolder(
                root=directory,
                transform=self.transform)

        if balanced_sampling:

            # balanced sampler
            counts = np.bincount(dataset.targets)
            labels_weights = 1. / counts
            weights = labels_weights[dataset.targets]
            ws = WeightedRandomSampler(weights, len(weights), replacement=True)

            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    drop_last=True,
                    num_workers=self.num_workers,
                    sampler=ws)
        else:
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers)

        return dataset, dataloader

    def tensor_label(
            self,
            directory: str,
            label_name: str,
            ) -> torch.Tensor:

        dataset, dataloader = self(directory)

        label_idx = dataset.class_to_idx[label_name]
        label_idx_list = [
                idx for idx, target_idx in enumerate(dataset.targets)
                if target_idx == label_idx]

        label_dataset = dataset[label_idx_list[0]][0].unsqueeze(0)
        for idx in label_idx_list[1:]:
            label_dataset = torch.cat(
                    (label_dataset, dataset[idx][0].unsqueeze(0)),
                    dim=0)

        return label_dataset

    def tensor(
            self,
            directory: str):

        dataset, dataloader = self(directory)

        total_targets = dataset.targets

        total_dataset = dataset[0][0].unsqueeze(0)
        for data_idx in range(1, len(dataset)):
            total_dataset = torch.cat(
                    (total_dataset, dataset[data_idx][0].unsqueeze(0)),
                    dim=0)

        return total_dataset, total_targets

    def numpy_label(
            self,
            directory: str,
            label_name: str,
            ) -> np.ndarray:

        label_dataset = self.tensor_label(
                directory=directory,
                label_name=label_name)

        return label_dataset.numpy()

    def filename(
            self,
            directory: str,
            label_name: str,
            ) -> List[str]:

        dataset, dataloader = self(directory)

        label_idx = dataset.class_to_idx[label_name]
        label_idx_list = [
                idx for idx, target_idx in enumerate(dataset.targets)
                if target_idx == label_idx]

        filename_list = []
        for idx in label_idx_list:
            filename_list.append(dataset.imgs[idx][0])

        return filename_list


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Augmentation
    agmt = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomAffine(
                degrees=(30, 70),
                translate=(0.1, 0.3),
                scale=(0.5, 0.75)),
            ]
    agmt_names = [
            'Horizontal Flip',
            'Vertical Flip',
            'Rotation',
            'Affine',
            ]

    transformer = [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            ]

    fig, axes = plt.subplots(len(agmt) + 1, 4)

    # Load Image
    Loader = Load(transformer)
    dataset, _ = Loader("final_dataset\\train\\")
    origin_images = [dataset[i][0] for i in range(4)]

    images = origin_images
    image = images[0].numpy().transpose(1, 2, 0)
    axes[0][0].imshow(image)
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])
    axes[0][0].set_ylabel('original', fontsize=8)
    image = images[1].numpy().transpose(1, 2, 0)
    axes[0][1].imshow(image)
    axes[0][1].set_xticks([])
    axes[0][1].set_yticks([])
    image = images[2].numpy().transpose(1, 2, 0)
    axes[0][2].imshow(image)
    axes[0][2].set_xticks([])
    axes[0][2].set_yticks([])
    image = images[3].numpy().transpose(1, 2, 0)
    axes[0][3].imshow(image)
    axes[0][3].set_xticks([])
    axes[0][3].set_yticks([])

    for idx, transform in enumerate(agmt):
        images = [transform(image) for image in origin_images]

        image = images[0].numpy().transpose(1, 2, 0)
        axes[idx+1][0].imshow(image)
        axes[idx+1][0].set_xticks([])
        axes[idx+1][0].set_yticks([])
        axes[idx+1][0].set_ylabel(
                agmt_names[idx],
                fontsize=8)
        image = images[1].numpy().transpose(1, 2, 0)
        axes[idx+1][1].imshow(image)
        axes[idx+1][1].set_xticks([])
        axes[idx+1][1].set_yticks([])
        image = images[2].numpy().transpose(1, 2, 0)
        axes[idx+1][2].imshow(image)
        axes[idx+1][2].set_xticks([])
        axes[idx+1][2].set_yticks([])
        image = images[3].numpy().transpose(1, 2, 0)
        axes[idx+1][3].imshow(image)
        axes[idx+1][3].set_xticks([])
        axes[idx+1][3].set_yticks([])

    plt.show()
