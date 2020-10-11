import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, Lambda

from data import CaptchaDataset
from model import CaptchaModel, CaptchaModelFixedSize
from util import naive_ctc_decode


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
        for images, labels in dataloader:
            if phase == 'train':
                optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            # batch = output.shape[1]
            # width = output.shape[0]
            # input_lens = torch.full(size=(batch,), fill_value=width, dtype=torch.long, device=device)
            # label_size = labels.shape[1]
            # label_lens = torch.full(size=(batch,), fill_value=label_size, dtype=torch.long, device=device)
            # loss = criterion(output, labels, input_lens, label_lens)
            loss = criterion(output, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            batch = labels.shape[0]
            total += batch
            total_loss += loss.item()

            labels = labels.detach().cpu().numpy()
            labels = [labels[i] for i in range(batch)]
            output = output.detach().cpu().numpy()
            # output = [output[:, i, :] for i in range(batch)]
            # output = [naive_ctc_decode(o) for o in output]
            output = output.argmax(axis=1)
            output = [output[i] for i in range(batch)]
            total_acc += sum(np.array_equal(o, l) for o, l in zip(output, labels))

    return total_loss / total, total_acc / total


def train(train_dir, ext, train_ratio, device, nworkers, max_epochs, modelname, checkpoint, lr):
    train_dir = Path(train_dir)
    assert train_dir.is_dir()
    train_paths = list(train_dir.glob('*.{}'.format(ext)))
    assert len(train_paths) > 0

    label_size = 5

    # train set containts only 5-letter labels
    train_paths = [p for p in train_paths if len(p.stem) == label_size]

    seed = 1
    random.seed(seed)
    random.shuffle(train_paths)
    assert 0 < train_ratio < 1
    train_split = int(len(train_paths) * train_ratio)
    train_paths, valid_paths = train_paths[:train_split], train_paths[train_split:]

    preload = True
    height = 120
    width = 300
    rmin, rmax, cmin, cmax = 10, 129, 8, 307
    assert rmax - rmin + 1 == height
    assert cmax - cmin + 1 == width
    mean = (0.5461, 0.6021, 0.6530)
    std = (0.3666, 0.3641, 0.3365)

    # train_transform = Compose([
    #     RandomApply([RandomChoice([
    #         Grayscale(num_output_channels=3),
    #         ColorJitter(hue=0.5),
    #     ])], p=0.4),
    #     Resize(size=(height, width)),
    #     ToTensor(),
    #     Normalize(mean, std, inplace=True),
    # ])

    valid_transform = Compose([
        # Resize(size=(height, width)),
        ToTensor(),
        Lambda(lambda x: x[:, rmin:rmax+1, cmin:cmax+1]),
        Normalize(mean, std, inplace=True),
    ])

    train_dataset = CaptchaDataset(
        paths=train_paths,
        with_labels=True,
        preload=preload,
        transform=valid_transform,
    )
    valid_dataset = CaptchaDataset(
        paths=valid_paths,
        with_labels=True,
        preload=preload,
        transform=valid_transform,
    )

    batch_size = 32

    train_iter = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=nworkers)
    valid_iter = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=nworkers)

    # model = CaptchaModel(len(train_dataset.symbols)).to(device)
    # criterion = nn.CTCLoss(zero_infinity=True).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=1e-1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    model = CaptchaModelFixedSize(len(train_dataset.symbols), label_size).to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    start_epoch = 1

    best_loss = 1e6
    best_path = None
    best_acc = None

    if checkpoint is not None:
        assert Path(checkpoint).is_file()
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        best_loss = checkpoint['best_loss']
        best_acc = checkpoint['best_acc']

    writer = SummaryWriter('logdir/{}'.format(modelname))
    sample_input = next(iter(train_iter))
    sample_images, sample_labels = sample_input
    for image, label in zip(sample_images, sample_labels):
        label = ''.join(train_dataset.itos(i.item()) for i in label)
        writer.add_image('Sample/{}'.format(label), image)
    with torch.no_grad():
        writer.add_graph(model, sample_images.to(device))
    writer.flush()

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss, train_acc = run_one_epoch('train', train_iter, model, criterion, optimizer, device)
        valid_loss, valid_acc = run_one_epoch('valid', valid_iter, model, criterion, None, device)
        writer.add_scalar('Loss/Training', train_loss, epoch)
        writer.add_scalar('Loss/Validation', valid_loss, epoch)
        writer.add_scalar('Accuracy/Training', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', valid_acc, epoch)
        # scheduler.step(valid_loss)
        print('Epoch: {}'.format(epoch))
        print('Train loss: {:.3f}\tTrain acc: {:.3f}'.format(train_loss, train_acc))
        print('Valid loss: {:.3f}\tValid acc: {:.3f}'.format(valid_loss, valid_acc))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = valid_acc
            old_path = best_path
            best_path = 'captcha_epoch_{:03d}_acc_{:.3f}.pt'.format(epoch, valid_acc)
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc,
            }
            torch.save(checkpoint, best_path)
            if old_path is not None and Path(old_path).is_file():
                Path(old_path).unlink()


def launch_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--ext', default='png')
    parser.add_argument('--train_ratio', default=0.8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--nworkers', default=0, type=int)
    parser.add_argument('--max_epochs', default=10000, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--lr', default='1e-3', type=str)
    args = parser.parse_args()

    train(args.train_dir, args.ext, args.train_ratio, args.device, args.nworkers, args.max_epochs, args.modelname,
          args.checkpoint, args.lr)


if __name__ == '__main__':
    launch_train()
