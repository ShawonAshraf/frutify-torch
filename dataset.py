import os
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io

from image_utils import generate_labels, get_label_from_file_name
from image_transforms import Rescale, RotateImage, RandomCrop, ToTensor

IMAGE_FOLDER_NAME = "dataset"
IMAGE_DIR_PATH = os.path.join(os.getcwd(), IMAGE_FOLDER_NAME)

FRUIT_NAMES = ["apple", "orange", "banana", "mango"]


class FruitImageDataset(Dataset):
    """
    set test to true for test datasets
    and False for training/validation datasets

    image_file_names should contain the list of images
    you want to include in the dataset (test / validation / train)
    """

    def __init__(self,
                 image_file_names,
                 test,
                 image_height=300,
                 image_width=300,
                 dataset_path=IMAGE_DIR_PATH):
        self.labels = generate_labels(FRUIT_NAMES)

        self.h = image_height
        self.w = image_width
        self.dataset_path = dataset_path

        self.image_file_names = image_file_names

        # compose transforms
        if not test:
            # for train run all the defined transforms
            self.transform = transforms.Compose([
                Rescale((self.h, self.w)),
                RandomCrop(224),  # for inception, 224
                RotateImage(-90.0),
                RotateImage(90.0),
                RotateImage(-180.0),
                RotateImage(180.0),
                ToTensor()
            ])
        else:
            # for testing apply only rescale and totensor
            self.transform = transforms.Compose([
                Rescale((self.h, self.w)),
                ToTensor()
            ])

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
            "label": self.labels.index(label),  # index as label
            "file_name": image_name
        }

        # apply transforms
        item = self.transform(item)

        return item
