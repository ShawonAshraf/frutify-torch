import os
import math
import random

import torch.utils.data as D

IMAGE_FOLDER_NAME = "dataset"
IMAGE_DIR_ROOT = os.path.join(os.getcwd(), IMAGE_FOLDER_NAME)

FRUIT_NAMES = ["apple", "orange", "banana", "mango"]


def generate_labels(fruit_names=FRUIT_NAMES):
    labels = []

    for fruit_name in fruit_names:
        labels.append("fresh_" + fruit_name)
        labels.append("rotten_" + fruit_name)

    return labels


def get_label_from_file_name(file_name):
    splits = file_name.split("_")
    # file name example: fresh_orange_1.jpg
    # the label is the first two tokens joined by _
    label = "_".join(s for s in splits[:-1])
    return label


def get_all_file_names(root_dir=IMAGE_DIR_ROOT):
    return os.listdir(root_dir)


# train_ratio is a float number
# https://stackoverflow.com/a/61818182/3316525

# returns 3 lists, containing image names for train, validation and test sets
def create_train_test_dev_set(all_file_names, train_ratio):
    total_len = len(all_file_names)

    train_len = int(train_ratio * total_len)
    val_len = int((1 - train_ratio) * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = D.random_split(
        all_file_names, (train_len, val_len, test_len)
    )

    # random split returns a torch dataset
    # but strings are required here!
    train_set = [str(i) for i in train_set]
    test_set = [str(i) for i in test_set]
    val_set = [str(i) for i in val_set]

    return train_set, val_set, test_set
