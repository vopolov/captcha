import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Grayscale, ColorJitter, RandomChoice, RandomApply
from torchvision.transforms import Resize, ToTensor, Normalize, Compose

from data import CaptchaDataset
from model import CaptchaModel
from util import naive_ctc_decode

torch.backends.cudnn.enabled = False


def run_one_epoch(phase, dataloader, model, criterion, optimizer, device):
    assert phase in ['train', 'valid']
    if phase == 'train':
        model.train()
    elif phase == 'valid':
        model.eval()
    else:
        return

    total_loss = 0
    total_acc = 0
    total = 0

    with torch.set_grad_enabled(phase == 'train'):
        for images, labels, label_lens in dataloader:
            if phase == 'train':
                optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

            output = model(images)

            batch = output.shape[1]
            width = output.shape[0]
            input_lens = torch.full(size=(batch,), fill_value=width, dtype=torch.long, device=device)
            loss = criterion(output, labels, input_lens, label_lens)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            total += batch
            total_loss += loss.item()

            labels = labels.detach().cpu().numpy()
            labels = [labels[i] for i in range(batch)]
            output = output.detach().cpu().numpy()
            output = [output[:, i, :] for i in range(batch)]
            output = [naive_ctc_decode(o) for o in output]
            total_acc += sum(np.array_equal(o, l) for o, l in zip(output, labels))

    return total_loss / total, total_acc / total


def train(train_dir, ext, train_ratio, device, nworkers, max_epochs, checkpoint=None):
    train_dir = Path(train_dir)
    assert train_dir.is_dir()
    train_paths = list(train_dir.glob('*.{}'.format(ext)))
    assert len(train_paths) > 0

    seed = 1
    random.seed(seed)
    random.shuffle(train_paths)
    assert 0 < train_ratio < 1
    train_split = int(len(train_paths) * train_ratio)
    train_paths, valid_paths = train_paths[:train_split], train_paths[train_split:]

    preload = True
    height = 150
    width = 330
    mean = (171.28, 181.15, 190.28)
    std = (94.81, 91.23, 83.4)

    train_transform = Compose([
        RandomApply([RandomChoice([
            Grayscale(num_output_channels=3),
            ColorJitter(hue=0.5),
        ])], p=0.4),
        Resize(size=(height, width)),
        ToTensor(),
        Normalize(mean, std, inplace=True),
    ])

    valid_transform = Compose([
        Resize(size=(height, width)),
        ToTensor(),
        Normalize(mean, std, inplace=True),
    ])

    train_dataset = CaptchaDataset(
        paths=train_paths,
        with_labels=True,
        preload=preload,
        transform=train_transform,
    )
    valid_dataset = CaptchaDataset(
        paths=valid_paths,
        with_labels=True,
        preload=preload,
        transform=valid_transform,
    )

    batch_size = 16

    def captcha_collate(batch):
        images, labels = zip(*batch)
        label_lens = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)
        images = torch.stack(images)
        labels = torch.cat(labels)
        return images, labels, label_lens

    train_iter = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nworkers, collate_fn=captcha_collate)
    valid_iter = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=nworkers, collate_fn=captcha_collate)

    model = CaptchaModel(len(train_dataset.symbols)).to(device)
    criterion = nn.CTCLoss(zero_infinity=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    start_epoch = 1

    if checkpoint is not None:
        assert Path(checkpoint).is_file()
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    best_loss = 1e6
    best_path = None
    best_acc = None

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss, train_acc = run_one_epoch('train', train_iter, model, criterion, optimizer, device)
        valid_loss, valid_acc = run_one_epoch('valid', valid_iter, model, criterion, None, device)
        scheduler.step(valid_loss)
        print('Epoch: {}'.format(epoch))
        print('Train loss: {:.3f}\tTrain acc: {:.3f}'.format(train_loss, train_acc))
        print('Valid loss: {:.3f}\tValid acc: {:.3f}'.format(valid_loss, valid_acc))
        if valid_loss < best_loss:
            best_loss = valid_loss
            old_path = best_path
            best_path = 'captcha_epoch_{:03d}_acc_{:.3f}.pt'.format(epoch, valid_acc)
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, best_path)
            if old_path is not None and Path(old_path).is_file():
                Path(old_path).unlink()


def launch_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--ext', default='png', choices=['png', 'jpg'])
    parser.add_argument('--train_ratio', default=0.8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--nworkers', default=4, type=int)
    parser.add_argument('--max_epochs', default=10000, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    args = parser.parse_args()

    train(args.train_dir, args.ext, args.train_ratio, args.device, args.nworkers, args.max_epochs, args.checkpoint)


if __name__ == '__main__':
    launch_train()
