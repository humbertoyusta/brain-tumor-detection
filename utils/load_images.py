import os
import cv2
import numpy as np


def load_images(data_root_folder="data", image_size=(224, 224)):
    """Load the given images of brain MRI scans and their labels.
    Returns a tuple of two numpy arrays: (images, labels).
    images: numpy array of shape (num_images, image_height, image_width, num_channels) (RGB images)
    labels: numpy array of shape (num_images,) (0 or 1)"""
    images = []
    labels = []

    data_folder = os.path.join(data_root_folder, "brain_mri_scan_images")

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

    return np.array(images), np.array(labels)
