import sys
import os
import datetime
import argparse

from image_utils import *
from dataset import FruitImageDataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from classfier import FrutifyResnet101, FrutifyInceptionV3

if __name__ == "__main__":
    # cmd argparse
    arg_parser = argparse.ArgumentParser()

    # add options for argparse
    arg_parser.add_argument("--model", type=str, required=True, help="resnet101 or inception-v3")
    # if GPU is supplied without number of gpus to use, the first gpu in the system will be used
    arg_parser.add_argument("--device", type=str, required=True, help="gpu or cpu")
    arg_parser.add_argument("--n_gpus", type=int)
    # number of workers to use for dataloader
    arg_parser.add_argument("--num_workers", type=int)
    # split ratio
    arg_parser.add_argument("--split", type=float, default=0.8, required=True, help="train-test split ratio")

    # batch size
    arg_parser.add_argument("--batch_size", type=int, required=True)
    # epochs
    arg_parser.add_argument("--epochs", required=True, type=int)
    # learning_rate
    arg_parser.add_argument("--lr", type=float, required=True)

    args = arg_parser.parse_args()

    # load data and create train test validation split
    all_file_names = get_all_file_names()
    t, v, ts = create_train_test_dev_set(all_file_names, args.split)

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

    if args.model == "resnet101":
        clf = FrutifyResnet101(n_classes, args.lr)
    else:
        clf = FrutifyInceptionV3(n_classes, args.lr)

    if args.device == "gpu":
        if args.n_gpus:
            trainer = pl.Trainer(gpus=args.n_gpus, max_epochs=args.epochs)
        else:
            trainer = pl.Trainer(gpus=1, max_epochs=args.epochs)

    else:
        trainer = pl.Trainer(max_epochs=args.epochs)

    # call trainer
    trainer.fit(clf, train_loader, validation_loader)

    # save trained model
    # create dir if doesn't exist
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

    saved_model_name = f"{args.model}_{args.epochs}_{args.batch_size}_{args.lr}_{datetime.datetime.now().timestamp()}.ckpt"
    trainer.save_checkpoint(os.path.join("saved_models", saved_model_name))


