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

if __name__ == "__main__":
    # cmd argparse
    arg_parser = argparse.ArgumentParser()

    # add options for argparse
    arg_parser.add_argument("--model", type=str, required=True, help="resnet101 or inception-v3")

    arg_parser.add_argument("--saved_path", type=str, required=True, help="path to the saved model")

    # number of workers to use for dataloader
    arg_parser.add_argument("--num_workers", type=int)

    # split ratio
    arg_parser.add_argument("--split", type=float, default=0.8, required=True, help="train-test split ratio")

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
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        # revert to default
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # =================================================
    # train
    # there are 8 classes in the dataset
    n_classes = 8

    # model path
    # sanitize for os file separator conventions
    model_path = os.path.abspath(args.saved_path)

    if args.model == "resnet101":
        clf = FrutifyResnet101.load_from_checkpoint(model_path)
    else:
        clf = FrutifyInceptionV3.load_from_checkpoint(model_path)

    # run test
    predicted = list()
    ground_truths = list()

    for idx, batch in enumerate(tqdm(test_loader), 0):
        image, ground_truth = batch["image"], batch["label"]
        ground_truths.extend(ground_truth.cpu().detach().numpy())

        pred = clf(image)
        # get the label index with the max probability
        pred = torch.argmax(pred, dim=1)

        predicted.extend(pred.cpu().detach().numpy())

    # classification report

    # get the target labels of the dataset
    target_labels = generate_labels()
    print(classification_report(y_true=ground_truths, y_pred=predicted, target_names=target_labels))

