import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from data import CaptchaDataset
from model import CaptchaModel
from util import naive_ctc_decode


def train_epoch(train_iter, model, criterion, optimizer):
    model.train()
    total_loss = 0
    total_acc = 0
    total = 0
    for images, labels in tqdm.tqdm(train_iter, 'Train', leave=True):
        optimizer.zero_grad()

        output = model(images)
        batch = output.shape[1]
        width = output.shape[0]
        input_lens = torch.full(size=(batch,), fill_value=width, dtype=torch.int32)
        label_size = labels.shape[1]
        target_lens = torch.full(size=(batch,), fill_value=label_size, dtype=torch.int32)
        labels = labels.view(-1).to('cpu')
        loss = criterion(output, labels, input_lens, target_lens)

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        total += batch
        total_loss += loss.item()
        labels = labels.detach().cpu().numpy()
        # labels = np.split(labels, label_size, axis=0)
        labels = [labels[s:f] for s, f in zip(range(0, len(labels), label_size),
                                              range(label_size, len(labels) + 1, label_size))]
        # labels = np.split(labels, batch, axis=1)
        output = output.detach().cpu().numpy()
        # output = np.split(output, batch, axis=1)
        output = [output[:, i, :] for i in range(batch)]
        output = [naive_ctc_decode(o) for o in output]
        total_acc += sum(np.array_equal(o, l) for o, l in zip(output, labels))

    return total_loss / total, total_acc / total


def train(train_dir, valid_dir, device, nworkers, epochs):
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

    train_dataset = CaptchaDataset(train_dir, device, preload, height, width, mean, std, label_size)
    valid_dataset = CaptchaDataset(valid_dir, device, preload, height, width, mean, std, label_size)

    batch_size = 32

    # def collate(batch):
    #     images, labels = zip(*batch)
    #     return torch.stack(images), torch.cat(labels)

    mp.set_start_method("spawn")
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nworkers)  #, collate_fn=collate)
    valid_iter = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=nworkers)  #, collate_fn=collate)

    model = CaptchaModel(train_dataset.dims, len(train_dataset.symbols), height).to(device)
    criterion = nn.CTCLoss().to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(train_iter, model, criterion, optimizer)
        print('Epoch: {}\nTrain loss:{}\tTrain acc:{}'.format(epoch, train_loss, train_acc))


def launch_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--valid_dir', default='valid')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--nworkers', default=4, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    args = parser.parse_args()

    train(args.train_dir, args.valid_dir, args.device, args.nworkers, args.epochs)


if __name__ == '__main__':
    launch_train()
