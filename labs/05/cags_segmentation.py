#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2
import torchmetrics
import npfl138
import npfl138.datasets.cags
npfl138.require_version("2425.5")
import npfl138.datasets
from npfl138.datasets.cags import CAGS
from npfl138 import TrainableModule
from torch.utils.data import DataLoader

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")

class CAGSSegmentationModel(TrainableModule):
    def __init__(self):
        super().__init__()
        
        
        self.encoder = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

       
        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Conv2d(1280, 512, kernel_size=3, padding=1),  # 7x7
        #     torch.nn.ReLU(),
        #     torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 7 -> 14

        #     torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 14 -> 28

        #     torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 28 -> 56

        #     torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),  # 56 -> 224

        #     torch.nn.Conv2d(64, 1, kernel_size=1),  # Output 1 channel (mask)
        # )
        self.decoder = DecoderWithSkipConnections()

    def forward(self, x):
        output,features = self.encoder.forward_intermediates(x)
        # output = self.encoder(x)
        return self.decoder(output, features)
class DecoderWithSkipConnections(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 7x7 14x14
        self.up1 = torch.nn.Sequential(
            torch.nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.conv1 = torch.nn.Conv2d(512 + 192, 512, kernel_size=3, padding=1)  # Merge with f5

        # 14x14 28x28
        self.up2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.conv2 = torch.nn.Conv2d(256 + 112, 256, kernel_size=3, padding=1)  # Merge with f4

        # 28x28  56x56
        self.up3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.conv3 = torch.nn.Conv2d(128 + 48, 128, kernel_size=3, padding=1)  # Merge with f3

        # 56x56 112x112
        self.up4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.conv4 = torch.nn.Conv2d(64 + 32, 64, kernel_size=3, padding=1)  # Merge with f1

        # 112x112 -> 224x224 (final resolution)
        self.up5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.final_conv = torch.nn.Conv2d(32, 1, kernel_size=1)  # Output 1 channel (mask)

    def forward(self, x, features):
        f1, f2, f3, f4, f5 = features  # Unpacking intermediate features

        # Upsample from the deepest feature map
        x = self.up1(x)
        f5_resized = torch.nn.functional.interpolate(f5, size=x.shape[2:], mode='bilinear', align_corners=True)  
        x = torch.cat([x, f5_resized], dim=1)  # Skip connection
        x = self.conv1(x)

        x = self.up2(x)
        f4_resized = torch.nn.functional.interpolate(f4, size=x.shape[2:], mode='bilinear', align_corners=True) 
        x = torch.cat([x, f4_resized], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        f3_resized = torch.nn.functional.interpolate(f3, size=x.shape[2:], mode='bilinear', align_corners=True) 
        x = torch.cat([x, f3_resized], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        f2_resized = torch.nn.functional.interpolate(f2, size=x.shape[2:], mode='bilinear', align_corners=True)  
        x = torch.cat([x, f2_resized], dim=1)
        x = self.conv4(x)

        x = self.up5(x)
        x = self.final_conv(x)

        return x
class ManualDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: CAGS.Dataset, preprocess, augmentation_fn=None) -> None:
        self._dataset = dataset
        self._augmentation_fn = augmentation_fn
        self._preprocess = preprocess

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._dataset[index]["image"]
        mask = self._dataset[index]["mask"]
        # label = self._dataset[index]["label"]
        image = self._preprocess(image)
        image = image.to(torch.float32) / 255
        if self._augmentation_fn is not None:
            image = self._augmentation_fn(image)
            mask = self._augmentation_fn(mask)
        return (image, mask)
def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
    augment = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    v2.RandomRotation(degrees=15),
    v2.RandomResizedCrop(size=224, scale=(0.9, 1.1), ratio=(0.90, 1.10)),
    v2.ElasticTransform(alpha=20.0)

])

    # TODO: Create the model and train it.
    model = CAGSSegmentationModel()

    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.0001),
        loss=torch.nn.BCEWithLogitsLoss(),
        metrics= {"MaskIoUMetric":npfl138.datasets.cags.CAGS.MaskIoUMetric()}
    )
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    train = ManualDataset(cags.train,
                        preprocess=preprocessing, augmentation_fn=augment)
    test = ManualDataset(cags.test,
                        preprocess=preprocessing,augmentation_fn=augment)
    dev = ManualDataset(cags.dev, 
                        preprocess=preprocessing,augmentation_fn=augment)

    train = DataLoader(train, batch_size=args.batch_size, shuffle=True,num_workers=args.dataloader_workers,persistent_workers=args.dataloader_workers > 0)
    dev = DataLoader(dev, batch_size=args.batch_size, shuffle=False,num_workers=args.dataloader_workers,persistent_workers=args.dataloader_workers > 0)
    test = DataLoader(test, batch_size=args.batch_size, shuffle=False,num_workers=args.dataloader_workers,persistent_workers=args.dataloader_workers > 0)
    model.fit(train, dev=dev, epochs=args.epochs, log_graph=True)
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for mask in model.predict(test, data_with_labels=True):
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
