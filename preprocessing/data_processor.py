import torch
import torch.utils.data
import numpy as np
import albumentations
import albumentations.pytorch
import preprocessing.crop_brain_region
import preprocessing.constants
import preprocessing.dataset
from typing import Tuple


class DataProcessor:
    def __init__(
        self,
        train_images: np.ndarray,
        val_images: np.ndarray,
        test_images: np.ndarray,
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        test_labels: np.ndarray,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        self.train_images = train_images
        self.val_images = val_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _define_transforms(self) -> None:

        self.test_transform = albumentations.Compose(
            [
                preprocessing.crop_brain_region.CropBrainRegion(
                    output_size=preprocessing.constants.IMAGE_SIZE
                ),
                albumentations.Normalize(
                    mean=preprocessing.constants.MEAN, std=preprocessing.constants.STD
                ),
                albumentations.pytorch.ToTensorV2(),
            ]
        )

        self.train_transform = albumentations.Compose(
            [
                preprocessing.crop_brain_region.CropBrainRegion(
                    output_size=preprocessing.constants.IMAGE_SIZE
                ),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5
                ),
                albumentations.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50, p=0.5
                ),
                albumentations.GaussNoise(var_limit=(10, 50), p=0.5),
                albumentations.Normalize(
                    mean=preprocessing.constants.MEAN, std=preprocessing.constants.STD
                ),
                albumentations.pytorch.ToTensorV2(),
            ]
        )

    def _create_datasets(self) -> None:
        self.train_dataset = preprocessing.dataset.BrainTumorDataset(
            self.train_images, self.train_labels, transform=self.train_transform
        )
        self.val_dataset = preprocessing.dataset.BrainTumorDataset(
            self.val_images, self.val_labels, transform=self.test_transform
        )
        self.test_dataset = preprocessing.dataset.BrainTumorDataset(
            self.test_images, self.test_labels, transform=self.test_transform
        )

    def _create_data_loaders(self) -> None:
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def run(
        self,
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        self._define_transforms()
        self._create_datasets()
        self._create_data_loaders()

        return self.train_loader, self.val_loader, self.test_loader
