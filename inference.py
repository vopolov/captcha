import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

from data import CaptchaDataset
from model import CaptchaModelFixedSize


def launch_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--weights', default='weights.pt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ext', default='png')
    parser.add_argument('--save_preds', default='preds.json')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--nworkers', default=0, type=int)
    parser.add_argument('--label_size', default=5, type=int)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.is_dir()
    assert Path(args.weights).is_file()
    paths = list(data_dir.glob('*.{}'.format(args.ext)))

    preload = False
    height = 120
    width = 300
    rmin, rmax, cmin, cmax = 10, 129, 8, 307
    assert rmax - rmin + 1 == height
    assert cmax - cmin + 1 == width
    mean = (0.5461, 0.6021, 0.6530)
    std = (0.3666, 0.3641, 0.3365)

    transform = Compose([
        ToTensor(),
        Lambda(lambda x: x[:, rmin:rmax+1, cmin:cmax+1]),
        Normalize(mean, std, inplace=True),
    ])

    dataset = CaptchaDataset(paths=paths, with_labels=True, preload=preload, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers)

    model = CaptchaModelFixedSize(len(dataset.symbols), args.label_size).to(args.device)
    model.load_state_dict(torch.load(Path(args.weights)))
    model.eval()

    total = 0
    total_acc = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            output = model(images)
            output = output.argmax(dim=1)
            output = output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            batch = labels.shape[0]
            total += batch

            output = list(output)
            labels = list(labels)

            total_acc += sum(np.array_equal(o, l) for o, l in zip(output, labels))

    total_acc /= total
    print('Per sample accuracy on {}:\n{:.2f} %'.format(args.data_dir, total_acc * 100))


if __name__ == '__main__':
    launch_inference()
