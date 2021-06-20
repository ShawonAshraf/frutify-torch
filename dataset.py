import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from skimage import io, transform
import numpy as np

from image_utils import generate_labels, get_label_from_file_name

IMAGE_FOLDER_NAME = "dataset"
IMAGE_DIR_PATH = os.path.join(os.getcwd(), IMAGE_FOLDER_NAME)

FRUIT_NAMES = ["apple", "orange", "banana", "mango"]


class FruitImageDataset(Dataset):
    def __init__(self,
                 image_height=224,
                 image_width=224,
                 dataset_path=IMAGE_DIR_PATH,
                 transform=None):
        self.labels = generate_labels(FRUIT_NAMES)

        self.h = image_height
        self.w = image_width
        self.dataset_path = dataset_path
        self.transform = transform

        self.image_file_names = os.listdir(self.dataset_path)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        image_name = self.image_file_names[idx]

        # imread requires full image path
        image = io.imread(
            os.path.join(
                IMAGE_DIR_PATH, image_name
            )
        )
        label = get_label_from_file_name(image_name)

        item = {
            "image": image,
            "label": label,
            "file_name": image_name
        }

        # apply transforms if mentioned
        if self.transform:
            item = self.transform(item)

        return item
