from typing import List

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

# Use my class
import sys
sys.path.append("../")
from load import Load


class agmt_previewer():

    def __init__(self):

        # Augmentation
        self.agmt = [
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomAffine(
                    degrees=(30, 70),
                    translate=(0.1, 0.3),
                    scale=(0.5, 0.75)),
                ]
        self.agmt_names = [
                'Horizontal Flip',
                'Vertical Flip',
                'Rotation',
                'Affine',
                ]

        self.transformer = [
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                ]

        self.fig, self.axes = plt.subplots(len(self.agmt) + 1, 4)

    def show_imgs(
            self,
            row: int,
            images: List[torch.Tensor],
            label: str):

        image = images[0].numpy().transpose(1, 2, 0)
        self.axes[row][0].imshow(image)
        self.axes[row][0].set_xticks([])
        self.axes[row][0].set_yticks([])
        self.axes[row][0].set_ylabel(label, fontsize=8)
        image = images[1].numpy().transpose(1, 2, 0)
        self.axes[row][1].imshow(image)
        self.axes[row][1].set_xticks([])
        self.axes[row][1].set_yticks([])
        image = images[2].numpy().transpose(1, 2, 0)
        self.axes[row][2].imshow(image)
        self.axes[row][2].set_xticks([])
        self.axes[row][2].set_yticks([])
        image = images[3].numpy().transpose(1, 2, 0)
        self.axes[row][3].imshow(image)
        self.axes[row][3].set_xticks([])
        self.axes[row][3].set_yticks([])

    def __call__(
            self,
            origin_images: List[torch.Tensor],
            ):

        self.show_imgs(0, origin_images, 'origin')

        for idx, transform in enumerate(self.agmt):
            images = [transform(image) for image in origin_images]
            self.show_imgs(idx+1, images, self.agmt_names[idx])

        plt.show()


if __name__ == "__main__":
    previewer = agmt_previewer()
    # Load Image
    Loader = Load(previewer.transformer)
    dataset, _ = Loader("..\\final_dataset\\train\\")
    origin_images = [dataset[i][0] for i in range(4)]

    previewer(origin_images)
