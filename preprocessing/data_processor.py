import os
import logging
import cv2
import imagededup.handlers
import imagededup.handlers.metrics
import imagededup.handlers.search
import imagededup.utils
import imagededup.utils.logger
import torch
import torch.utils.data
import numpy as np
import albumentations
import albumentations.pytorch
import preprocessing.crop_brain_region
import preprocessing.constants
import preprocessing.dataset
import imagededup
import imagededup.methods
import sklearn.model_selection
from typing import Tuple, List


class DataProcessor:
    def __init__(
        self,
        data_root_folder: str = "data",
        batch_size: int = 8,
        num_workers: int = 0,
        remove_duplicates: bool = True,
        log_level: int = logging.WARNING,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.remove_duplicates = remove_duplicates
        self.data_root_folder = data_root_folder
        self.raw_data_folder = os.path.join(data_root_folder, "raw")
        self.positive_images_count = {"train": 0, "val": 0, "test": 0}
        self.negative_images_count = {"train": 0, "val": 0, "test": 0}
        self.corrupted_images_count = 0
        self.log_level = log_level

    def _is_image_corrupted(self, filepath: str) -> bool:
        try:
            img = cv2.imread(filepath)
            if img is None:
                print(f"Corrupted image: {filepath}")
                return True
            return False
        except Exception as e:
            print(f"Corrupted image: {filepath}, Error: {e}")

    def _set_log_level(self, level: int) -> None:
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.WARNING)
            logger.propagate = False

    def _get_corrupted_images(self, filenames: List[str]) -> List[str]:
        self.corrupted_images = []

        for filename in filenames:
            if self._is_image_corrupted(os.path.join(self.raw_data_folder, filename)):
                self.corrupted_images.append(filename)

        return self.corrupted_images

    def _get_duplicates_to_remove(self) -> List[str]:

        phasher = imagededup.methods.PHash(verbose=False)
        encodings = phasher.encode_images(image_dir=self.raw_data_folder)

        # Find pairs of duplicates, for visualization purposes
        duplicates = phasher.find_duplicates(
            encoding_map=encodings, max_distance_threshold=0
        )

        self.duplicated_pairs = []
        for k, v in duplicates.items():
            for img in v:
                if k < img:
                    self.duplicated_pairs.append((k, img))

        # Find duplicates to remove
        self.duplicates_to_remove = phasher.find_duplicates_to_remove(
            encoding_map=encodings, max_distance_threshold=0
        )

        return self.duplicates_to_remove

    def _assert_images_sizes_and_channels(self, images: List[np.ndarray]) -> None:
        """Asserts that all images have the same size and number of channels."""

        for i, img in enumerate(images):
            assert img.shape == preprocessing.constants.IMAGE_SIZE + (3,)
            assert img.dtype == np.uint8

            if i > 0:
                assert img.shape == images[i - 1].shape

        print("All images have the same size and number of channels.")

    def _load_images(self, filenames: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        self.images = []
        self.labels = []
        for filename in filenames:
            img = cv2.imread(os.path.join(self.raw_data_folder, filename))
            img = cv2.resize(img, preprocessing.constants.IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img)
            self.labels.append(int(filename.split("_")[0]))

        self._assert_images_sizes_and_channels(self.images)
        return self.images, self.labels

    def _save_image(self, img: np.ndarray, label: int, folder: str) -> None:
        if label == 0:
            img_name = f"0_{self.negative_images_count[folder]}"
            self.negative_images_count[folder] += 1
        else:
            img_name = f"1_{self.positive_images_count[folder]}"
            self.positive_images_count[folder] += 1

        folder = os.path.join(self.data_root_folder, folder)
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(
            os.path.join(folder, f"{img_name}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

    def _split_and_store(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        split_ratio: Tuple[float, float, float] = (0.75, 0.125, 0.125),
    ) -> None:
        self.train_images, remaining_images, self.train_labels, remaining_labels = (
            sklearn.model_selection.train_test_split(
                images,
                labels,
                test_size=split_ratio[1] + split_ratio[2],
                random_state=0,
            )
        )

        self.val_images, self.test_images, self.val_labels, self.test_labels = (
            sklearn.model_selection.train_test_split(
                remaining_images,
                remaining_labels,
                test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
                random_state=0,
            )
        )

        for img, label in zip(self.train_images, self.train_labels):
            self._save_image(img, label, "train")

        for img, label in zip(self.val_images, self.val_labels):
            self._save_image(img, label, "val")

        for img, label in zip(self.test_images, self.test_labels):
            self._save_image(img, label, "test")

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
        self._set_log_level(self.log_level)

        filenames = os.listdir(self.raw_data_folder)
        print("Initial number of images:", len(filenames))

        corrupted_images = self._get_corrupted_images(filenames)
        print("Number of corrupted images:", len(corrupted_images))

        for corrupted_image in corrupted_images:
            filenames.remove(corrupted_image)

        if self.remove_duplicates:
            duplicates_to_remove = self._get_duplicates_to_remove()
            duplicates_removed = 0

            for duplicate in duplicates_to_remove:
                if duplicate in filenames:
                    filenames.remove(duplicate)
                    duplicates_removed += 1

            print("Number of duplicates removed:", duplicates_removed)

        print(
            "Number of images after removing corrupted",
            "and duplicated" if self.remove_duplicates else "",
            "images:",
            len(filenames),
        )

        images, labels = self._load_images(filenames)
        self._split_and_store(images, labels)
        self._define_transforms()
        self._create_datasets()
        self._create_data_loaders()

        return self.train_loader, self.val_loader, self.test_loader
