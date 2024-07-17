import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG16(nn.Module):

    def __init__(
            self,
            inputs: tuple = (3, 224, 224),
            outputs: int = 20):

        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.flatten = nn.Flatten(1, 3)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=outputs, bias=True),
        )

    # Input as 224x224x3 image
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


class VGG16_v2(nn.Module):

    def __init__(
            self,
            inputs: tuple = (3, 224, 224),
            outputs: int = 20):

        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.flatten = nn.Flatten(1, 3)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=outputs, bias=True),
        )

    # Input as 224x224x3 image
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int):

        super().__init__()

        if input_size == output_size:
            self.identity_block1 = nn.Sequential()
        else:
            self.identity_block1 = self.identity_block(input_size, output_size)

        self.identity_block2 = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)
        self.basic_block1 = self.basic_block(input_size, output_size)
        self.basic_block2 = self.basic_block(output_size, output_size)

    def identity_block(
            self,
            input_size: int,
            output_size: int):

        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

    def basic_block(
            self,
            input_size: int,
            output_size: int):

        stride_size = int(output_size / input_size)

        block = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=(3, 3), stride=(stride_size, stride_size), padding=(1, 1), bias=False),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size, output_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),)

        return block

    def forward(
            self,
            x: torch.Tensor):

        identity1 = self.identity_block1(x)
        x = self.basic_block1(x)
        x += identity1
        x = self.relu(x)

        identity2 = self.identity_block2(x)
        x = self.basic_block2(x)
        x += identity2
        x = self.relu(x)

        return x


class ResNet18(nn.Module):

    def __init__(
            self,
            inputs: tuple = (3, 224, 224),
            outputs: int = 20):

        super().__init__()

        self._first_block = self.input_block()
        self._residual_block1 = ResidualBlock(64, 64)
        self._residual_block2 = ResidualBlock(64, 128)
        self._residual_block3 = ResidualBlock(128, 256)
        self._residual_block4 = ResidualBlock(256, 512)
        self._last_block = self.output_block(512, outputs)

    def input_block(self):

        block = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                )

        return block


    def output_block(
            self,
            input_feature_size: int,
            output_feature_size: int):

        block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1, 3),
            nn.Linear(in_features=input_feature_size, out_features=output_feature_size, bias=True),)

        return block

    def forward(
            self,
            x: torch.Tensor):

        x = self._first_block(x)
        x = self._residual_block1(x)
        x = self._residual_block2(x)
        x = self._residual_block3(x)
        x = self._residual_block4(x)
        x = self._last_block(x)

        return x


class BottleneckBlock(nn.Module):

    def __init__(
            self,
            input_size: int,
            middle_size: int,
            output_size: int,
            identity_stride=None):

        super().__init__()

        if identity_stride:
            self.identity_block1 = self.identity_block(input_size, output_size, identity_stride)
            self.bottleneck_block1 = self.bottleneck_block(input_size, middle_size, output_size, identity_stride)
        else:
            self.identity_block1 = nn.Sequential()
            self.bottleneck_block1 = self.bottleneck_block(input_size, middle_size, output_size, 1)

        self.relu = nn.ReLU(inplace=True)

    def identity_block(
            self,
            input_size: int,
            output_size: int,
            stride: int):

        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=(1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

    def bottleneck_block(
            self,
            input_size: int,
            middle_size: int,
            output_size: int,
            stride_size: int = 1):

        block = nn.Sequential(
            nn.Conv2d(input_size, middle_size, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(middle_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_size, middle_size, kernel_size=(3, 3), stride=(stride_size, stride_size), padding=(1, 1), bias=False),
            nn.BatchNorm2d(middle_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_size, output_size, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),)

        return block

    def forward(
            self,
            x: torch.Tensor,
            ):

        identity1 = self.identity_block1(x)
        x = self.bottleneck_block1(x)
        x += identity1
        x = self.relu(x)

        return x


class ResNet50(nn.Module):

    def __init__(
            self,
            inputs: tuple = (3, 224, 224),
            outputs: int = 20):

        super().__init__()

        self._first_block = self.input_block()
        self._bottleneck_block1 = BottleneckBlock(64, 64, 256, 1)
        self._bottleneck_block2 = BottleneckBlock(256, 64, 256)
        self._bottleneck_block3 = BottleneckBlock(256, 64, 256)
        self._bottleneck_block4 = BottleneckBlock(256, 128, 512, 2)
        self._bottleneck_block5 = BottleneckBlock(512, 128, 512)
        self._bottleneck_block6 = BottleneckBlock(512, 128, 512)
        self._bottleneck_block7 = BottleneckBlock(512, 128, 512)
        self._bottleneck_block8 = BottleneckBlock(512, 256, 1024 ,2)
        self._bottleneck_block9 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block10 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block11 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block12 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block13 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block14 = BottleneckBlock(1024 , 512, 2048, 2)
        self._bottleneck_block15 = BottleneckBlock(2048 , 512, 2048)
        self._bottleneck_block16 = BottleneckBlock(2048 , 512, 2048)
        self._last_block = self.output_block(2048, outputs)

    def input_block(self):

        block = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                )

        return block

    def output_block(
            self,
            input_feature_size: int,
            output_feature_size: int):

        block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1, 3),
            nn.Linear(in_features=input_feature_size, out_features=output_feature_size, bias=True),)

        return block

    def forward(
            self,
            x: torch.Tensor):

        x = self._first_block(x)
        x = self._bottleneck_block1(x)
        x = self._bottleneck_block2(x)
        x = self._bottleneck_block3(x)
        x = self._bottleneck_block4(x)
        x = self._bottleneck_block5(x)
        x = self._bottleneck_block6(x)
        x = self._bottleneck_block7(x)
        x = self._bottleneck_block8(x)
        x = self._bottleneck_block9(x)
        x = self._bottleneck_block10(x)
        x = self._bottleneck_block11(x)
        x = self._bottleneck_block12(x)
        x = self._bottleneck_block13(x)
        x = self._bottleneck_block14(x)
        x = self._bottleneck_block15(x)
        x = self._bottleneck_block16(x)
        x = self._last_block(x)

        return x


class ResNet101(nn.Module):

    def __init__(
            self,
            inputs: tuple = (3, 224, 224),
            outputs: int = 20):

        super().__init__()

        self._first_block = self.input_block()
        self._bottleneck_block1 = BottleneckBlock(64, 64, 256, 1)
        self._bottleneck_block2 = BottleneckBlock(256, 64, 256)
        self._bottleneck_block3 = BottleneckBlock(256, 64, 256)
        self._bottleneck_block4 = BottleneckBlock(256, 128, 512, 2)
        self._bottleneck_block5 = BottleneckBlock(512, 128, 512)
        self._bottleneck_block6 = BottleneckBlock(512, 128, 512)
        self._bottleneck_block7 = BottleneckBlock(512, 128, 512)
        self._bottleneck_block8 = BottleneckBlock(512, 256, 1024 ,2)
        self._bottleneck_block9 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block10 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block11 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block12 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block13 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block14 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block15 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block16 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block17 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block18 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block19 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block20 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block21 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block22 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block23 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block24 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block25 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block26 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block27 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block28 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block29 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block30 = BottleneckBlock(1024, 256, 1024)
        self._bottleneck_block31 = BottleneckBlock(1024 , 512, 2048, 2)
        self._bottleneck_block32 = BottleneckBlock(2048 , 512, 2048)
        self._bottleneck_block33 = BottleneckBlock(2048 , 512, 2048)
        self._last_block = self.output_block(2048, outputs)

    def input_block(self):

        block = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                )

        return block


    def output_block(
            self,
            input_feature_size: int,
            output_feature_size: int):
            
        block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1, 3),
            nn.Linear(in_features=input_feature_size, out_features=output_feature_size, bias=True),)

        return block

    def forward(
            self,
            x: torch.Tensor):

        x = self._first_block(x)
        x = self._bottleneck_block1(x)
        x = self._bottleneck_block2(x)
        x = self._bottleneck_block3(x)
        x = self._bottleneck_block4(x)
        x = self._bottleneck_block5(x)
        x = self._bottleneck_block6(x)
        x = self._bottleneck_block7(x)
        x = self._bottleneck_block8(x)
        x = self._bottleneck_block9(x)
        x = self._bottleneck_block10(x)
        x = self._bottleneck_block11(x)
        x = self._bottleneck_block12(x)
        x = self._bottleneck_block13(x)
        x = self._bottleneck_block14(x)
        x = self._bottleneck_block15(x)
        x = self._bottleneck_block16(x)
        x = self._bottleneck_block17(x)
        x = self._bottleneck_block18(x)
        x = self._bottleneck_block19(x)
        x = self._bottleneck_block20(x)
        x = self._bottleneck_block21(x)
        x = self._bottleneck_block22(x)
        x = self._bottleneck_block23(x)
        x = self._bottleneck_block24(x)
        x = self._bottleneck_block25(x)
        x = self._bottleneck_block26(x)
        x = self._bottleneck_block27(x)
        x = self._bottleneck_block28(x)
        x = self._bottleneck_block29(x)
        x = self._bottleneck_block30(x)
        x = self._bottleneck_block31(x)
        x = self._bottleneck_block32(x)
        x = self._bottleneck_block33(x)
        x = self._last_block(x)

        return x


if __name__ == "__main__":

    from torchvision.models import resnet101
    from torchsummary import summary

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = resnet101()
    summary(model, (3, 224, 224), device=device.type)
