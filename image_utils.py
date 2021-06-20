import os

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
