import sys
import os
import argparse

from image_utils import *
from dataset import FruitImageDataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from classfier import FrutifyClassifier

if __name__ == "__main__":
    # cmd argparse
    arg_parser = argparse.ArgumentParser()

    # add options for argparse
    arg_parser.add_argument("--model", type=str, required=True, help="resnet101 or inception-v3")
    # if GPU is supplied without a gpu_id, the first gpu in the system will be used
    arg_parser.add_argument("--device", type=str, required=True, help="gpu or cpu")
    arg_parser.add_argument("--gpu_id", type=int)
    # number of workers to use for dataloader
    arg_parser.add_argument("--num_workers", type=int)
    # split ratio
    arg_parser.add_argument("--split", type=float, default=0.8, required=True, help="train-test split ratio")

    # batch size
    arg_parser.add_argument("--batch_size", type=int, required=True)
    # epochs
    arg_parser.add_argument("--epochs", required=True, type=int)
    # learning_rate
    arg_parser.add_argument("--lr", type=int, required=True)

    args = arg_parser.parse_args()

    # load data and create train test validation split
    all_file_names = get_all_file_names()
    t, v, ts = create_train_test_dev_set(all_file_names, 0.8)

    # datasets
    train_dataset = FruitImageDataset(IMAGE_DIR_ROOT, t)
    validation_dataset = FruitImageDataset(IMAGE_DIR_ROOT, v)
    test_dataset = FruitImageDataset(IMAGE_DIR_ROOT, ts)

    # data loaders
    if args.num_workers:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        # revert to default
        # if you're using comet.ml logging, use the default one
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # =================================================
    # train
    # there are 8 classes in the dataset
    n_classes = 8

    clf = FrutifyClassifier(n_classes, 1e-3)

    if args.device == "gpu":
        trainer = pl.Trainer(gpus=args.gpu_id, max_epochs=args.epochs)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs)

    # call trainer
    trainer.fit(clf, train_loader, validation_loader, test_loader)
