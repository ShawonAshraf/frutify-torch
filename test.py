import argparse

from tqdm import tqdm

from image_utils import *
from dataset import FruitImageDataset

import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import classification_report

import pytorch_lightning as pl
from classfier import FrutifyResnet101, FrutifyInceptionV3


"""
runs inference on the test set for a given model
"""


def run_inference(model, dataset):
    predicted = list()
    ground_truths = list()

    for idx, batch in enumerate(tqdm(test_loader), 0):
        image, ground_truth = batch["image"], batch["label"]
        ground_truths.extend(ground_truth.cpu().detach().numpy())

        pred = model(image)
        # get the label index with the max probability
        pred = torch.argmax(pred, dim=1)

        predicted.extend(pred.cpu().detach().numpy())

    return predicted, ground_truths


if __name__ == "__main__":
    # cmd argparse
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--inception_path", type=str, required=True,
                            help="path for the saved inception model")

    arg_parser.add_argument("--resnet_path", type=str,
                            required=True, help="path for the resnet model")

    # number of workers to use for dataloader
    arg_parser.add_argument("--num_workers", type=int)

    # split ratio
    arg_parser.add_argument("--split", type=float, default=0.8,
                            required=True, help="train-test split ratio")

    # batch size
    arg_parser.add_argument("--batch_size", type=int, required=True)

    args = arg_parser.parse_args()

    # load data and create train test validation split
    all_file_names = get_all_file_names()
    t, v, ts = create_train_test_dev_set(all_file_names, args.split)

    # dataset
    test_dataset = FruitImageDataset(IMAGE_DIR_ROOT, ts)

    # data loaders
    if args.num_workers:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        # revert to default
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # =================================================
    # there are 8 classes in the dataset
    n_classes = 8

    # init models
    saved_models = dict()
    saved_models["inception-v3"] = FrutifyInceptionV3.load_from_checkpoint(
        args.inception_path)

    saved_models["resnet101"] = FrutifyResnet101.load_from_checkpoint(
        args.resnet_path)

    # get the target labels of the dataset
    target_labels = generate_labels()

    # run test
    for model_name in saved_models.keys():
        print("*" * 25)
        print(f"{model_name}")
        print()
        predicted, ground_truths = run_inference(
            saved_models[model_name], test_dataset)

        # classification report
        print(classification_report(y_true=ground_truths,
                                    y_pred=predicted, target_names=target_labels))
        print("*" * 25)
        print()
