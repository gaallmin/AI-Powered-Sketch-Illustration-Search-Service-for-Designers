import torch


class Preprocess():

    def __init__(self):
        pass

    def __call__(
            data: torch.Tensor) -> torch.Tensor:

        transforms = torch.nn.Sequential(
            transforms.CenterCrop(10),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        return transforms
