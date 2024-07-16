import os
import cv2
import numpy as np
import kaggle
from typing import Tuple


class DataCollector:
    def __init__(self, data_root_folder: str = "data"):
        self.data_root_folder = data_root_folder

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

    def run(self, should_download=True) -> Tuple[np.ndarray, np.ndarray]:
        if should_download:
            self.download_data()
        return self.load_images()
