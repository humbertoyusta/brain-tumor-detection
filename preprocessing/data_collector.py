import os
import cv2
import numpy as np
import kaggle
import random
import sklearn
import preprocessing.constants
from typing import Tuple


class DataCollector:
    def __init__(self, data_root_folder: str = "data") -> None:
        self.data_root_folder = data_root_folder
        self.positive_images_count = 0
        self.negative_images_count = 0

    def download_data(self) -> None:
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

    def _load_image(self, filename: str) -> np.ndarray:
        img = cv2.imread(filename)
        img = cv2.resize(img, preprocessing.constants.IMAGE_SIZE)
        return img

    def load_images(self) -> Tuple[np.ndarray, np.ndarray]:
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
            images.append(self._load_image(os.path.join(pos_folder, img_name)))
            labels.append(1)

        for img_name in os.listdir(neg_folder):
            images.append(self._load_image(os.path.join(neg_folder, img_name)))
            labels.append(0)

        second_data_folder = os.path.join(self.data_root_folder, "brain_tumor_dataset")

        second_pos_folder = os.path.join(second_data_folder, "yes")
        second_neg_folder = os.path.join(second_data_folder, "no")

        for img_name in os.listdir(second_pos_folder):
            images.append(self._load_image(os.path.join(second_pos_folder, img_name)))
            labels.append(1)

        for img_name in os.listdir(second_neg_folder):
            images.append(self._load_image(os.path.join(second_neg_folder, img_name)))
            labels.append(0)

        return np.array(images), np.array(labels)

    def _save_image(self, img: np.ndarray, label: int) -> None:
        if label == 0:
            img_name = f"0_{self.negative_images_count}"
            self.negative_images_count += 1
        else:
            img_name = f"1_{self.positive_images_count}"
            self.positive_images_count += 1

        folder = os.path.join(self.data_root_folder, "raw")
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, f"{img_name}.png"), img)

    def save_images(self, images: np.ndarray, labels: np.ndarray) -> None:
        for img, label in zip(images, labels):
            self._save_image(img, label)

    def split_and_store(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        split_ratio: Tuple[float, float, float] = (0.75, 0.125, 0.125),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        for img, label in zip(train_images, train_labels):
            self._save_image(img, label, "train")

        for img, label in zip(val_images, val_labels):
            self._save_image(img, label, "val")

        for img, label in zip(test_images, test_labels):
            self._save_image(img, label, "test")

        return (
            train_images,
            val_images,
            test_images,
            train_labels,
            val_labels,
            test_labels,
        )

    def run(self, should_download=True) -> Tuple[np.ndarray, np.ndarray]:
        if should_download:
            self.download_data()
        images, labels = self.load_images()
        self.save_images(images, labels)
        return images, labels
