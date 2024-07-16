import os
import cv2
import numpy as np
import kaggle
import random
import sklearn
from typing import Tuple


class DataCollector:
    def __init__(self, data_root_folder: str = "data"):
        self.data_root_folder = data_root_folder
        self.positive_images_count = {
            "train": 0,
            "val": 0,
            "test": 0,
        }
        self.negative_images_count = {
            "train": 0,
            "val": 0,
            "test": 0,
        }
        random.seed(0)
        np.random.seed(0)
        sklearn.random.seed(0)

    def download_data(self):
        os.makedirs(self.data_root_folder, exist_ok=True)

        kaggle.api.authenticate()

        kaggle.api.dataset_download_files(
            "volodymyrpivoshenko/brain-mri-scan-images-tumor-detection",
            path=self.data_root_folder,
            unzip=True,
        )

        kaggle.api.dataset_download_files(
            "navoneel/brain-mri-images-for-brain-tumor-detection",
            path=self.data_root_folder,
            unzip=True,
        )

    def load_images(
        self, image_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load the given images of brain MRI scans and their labels.
        Returns a tuple of two numpy arrays: (images, labels).
        images: numpy array of shape (num_images, image_height, image_width, num_channels) (RGB images)
        labels: numpy array of shape (num_images,) (0 or 1)"""
        images = []
        labels = []

        data_folder = os.path.join(self.data_root_folder, "brain_mri_scan_images")

        pos_folder = os.path.join(data_folder, "positive")
        neg_folder = os.path.join(data_folder, "negative")

        for img_name in os.listdir(pos_folder):
            img = cv2.imread(os.path.join(pos_folder, img_name))
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(1)

        for img_name in os.listdir(neg_folder):
            img = cv2.imread(os.path.join(neg_folder, img_name))
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(0)

        second_data_folder = os.path.join(self.data_root_folder, "brain_tumor_dataset")

        second_pos_folder = os.path.join(second_data_folder, "yes")
        second_neg_folder = os.path.join(second_data_folder, "no")

        for img_name in os.listdir(second_pos_folder):
            img = cv2.imread(os.path.join(second_pos_folder, img_name))
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(1)

        for img_name in os.listdir(second_neg_folder):
            img = cv2.imread(os.path.join(second_neg_folder, img_name))
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(0)

        return np.array(images), np.array(labels)

    def _save_image(self, img, label, folder):
        if label == 0:
            img_name = self.negative_images_count[folder]
            self.negative_images_count[folder] += 1
            folder = os.path.join(self.data_root_folder, folder, "negative")
        else:
            img_name = self.positive_images_count[folder]
            self.positive_images_count[folder] += 1
            folder = os.path.join(self.data_root_folder, folder, "positive")

        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(
            os.path.join(folder, f"{img_name}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

    def split_and_store(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        split_ratio: Tuple[float, float, float] = (0.75, 0.125, 0.125),
    ):
        train_images, remaining_images, train_labels, remaining_labels = (
            sklearn.model_selection.train_test_split(
                images,
                labels,
                test_size=split_ratio[1] + split_ratio[2],
                random_state=0,
            )
        )

        val_images, test_images, val_labels, test_labels = (
            sklearn.model_selection.train_test_split(
                remaining_images,
                remaining_labels,
                test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
                random_state=0,
            )
        )

        for i, (img, label) in enumerate(zip(train_images, train_labels)):
            self._save_image(img, label, "train")

        for i, (img, label) in enumerate(zip(val_images, val_labels)):
            self._save_image(img, label, "val")

        for i, (img, label) in enumerate(zip(test_images, test_labels)):
            self._save_image(img, label, "test")

    def run(self, should_download=True) -> Tuple[np.ndarray, np.ndarray]:
        if should_download:
            self.download_data()
        images, labels = self.load_images()
        self.split_and_store(images, labels)
        return images, labels
