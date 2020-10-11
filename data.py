import random
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CaptchaDataset(Dataset):
    def __init__(self, paths, with_labels=False, preload=False, transform=None):
        super().__init__()

        self._paths = paths
        self.with_labels = with_labels
        self.symbols = ['-'] + list(ascii_uppercase) + [str(i) for i in range(10)]
        self._index_dict = {i: s for i, s in enumerate(self.symbols, 0)}  # 0 left for predicted blank symbol
        self._symbols_dict = {s: i for i, s in enumerate(self.symbols, 0)}

        if transform:
            self._transform = transform
        else:
            self._transform = ToTensor()

        self._data = None
        self.preloaded = preload
        if self.preloaded:
            self._preload_data()

    def itos(self, index):
        return self._index_dict[index]

    def stoi(self, symbol):
        return self._symbols_dict[symbol]

    def _preload_data(self):
        self._data = []
        for p in self._paths:
            self._data.append(Image.open(p))

    def __getitem__(self, index):
        if self.preloaded:
            img = self._data[index]
        else:
            img = Image.open(self._paths[index])
        img = self._transform(img)
        if self.with_labels:
            label = Path(self._paths[index]).stem.upper()
            # target dtype must by int32 to work with cudnn ctc loss
            label = torch.tensor([self.stoi(s) for s in label], dtype=torch.int32)
            return img, label
        else:
            return img

    def __len__(self):
        return len(self._paths)


if __name__ == '__main__':
    data = Path('train')
    if data.is_dir():
        ps = list(data.glob('*.png'))
        d = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = CaptchaDataset(ps, with_labels=True, preload=True)
        idx = random.randint(0, len(dataset))
        image, text = dataset[idx]
        print(len(dataset))
        print(len(dataset.symbols))

        image = np.array(image.permute(1, 2, 0).cpu())
        image = image - np.min(image, axis=(0, 1))
        image = image / np.ptp(image, axis=(0, 1)) * 255
        image = image.astype('uint8')
        image = Image.fromarray(image)
        text = ''.join(dataset.itos(s.item()) for s in text)
        print(text)
        image.show(title=text)
