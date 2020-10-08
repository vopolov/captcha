import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import CaptchaDataset
from model import CaptchaModel


def train(train_dir, valid_dir, nworkers):
    train_dir = Path(train_dir)
    assert train_dir.is_dir()
    valid_dir = Path(valid_dir)
    assert valid_dir.is_dir()

    preload = True
    height = 150
    width = 330
    mean = (171.28, 181.15, 190.28)
    std = (94.81, 91.23, 83.4)
    label_size = 5

    train_dataset = CaptchaDataset(train_dir, preload, height, width, mean, std, label_size)
    valid_dataset = CaptchaDataset(valid_dir, preload, height, width, mean, std, label_size)

    batch_size = 256

    def collate(batch):
        images, labels = batch
        return torch.stack(images), torch.cat(labels)

    train_iter = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nworkers, collate_fn=collate)
    valid_iter = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=nworkers, collate_fn=collate)




def launch_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--valid_dir', default='valid')
    parser.add_argument('--nworkers', default=4, type=int)
    args = parser.parse_args()

    train(args.train_dir, args.valid_dir, args.nworkers)


if __name__ == '__main__':
    launch_train()
