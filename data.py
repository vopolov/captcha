import random
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CaptchaDataset(Dataset):
    def __init__(self, data_dir, preload, height, width,
                 mean=None, std=None, label_size=None, ext='png', mode='rgb'):
        super().__init__()

        self._data_dir = Path(data_dir)
        assert self._data_dir.is_dir(), '{} is not a valid directory'.format(self._data_dir)
        self._paths = list(self._data_dir.glob('*.{}'.format(ext)))
        assert len(self._paths) > 0, 'No files with {} extension found in {}'.format(ext, self._data_dir)

        mode_dict = {'RGB': 'RGB', 'rgb': 'RGB', 'gray': 'L', 'GRAY': 'L', 'L': 'L'}
        assert mode in mode_dict, '{} mode not supported'.format(mode)
        self.mode = mode_dict[mode]
        self.dims = 3 if self.mode == 'RGB' else 1

        self.height = height
        self.width = width

        self.mean = mean
        if self.mean:
            self.mean = torch.tensor(mean)
            assert len(self.mean) <= self.dims
        self.std = std
        if self.std:
            self.std = torch.tensor(std)
            assert len(self.std) <= self.dims

        self.label_size = label_size
        if self.label_size is not None:
            self._paths = [p for p in self._paths if len(p.stem) == self.label_size]

        self._symbols = list(ascii_uppercase) + [str(i) for i in range(10)]
        self._index_dict = {i: s for i, s in enumerate(self._symbols, 1)}  # 0 left for blank predicted symbol
        self._symbols_dict = {s: i for i, s in enumerate(self._symbols, 1)}

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
            self._data.append(self._read_image(p))

    def _read_image(self, path):
        img = Image.open(path)
        if img.mode != self.mode:
            img.convert(self.mode)
        if (self.height is not None and img.size[1] != self.height)\
                or (self.width is not None and img.size[0] != self.width):
            img = img.resize((self.width, self.height))

        img = np.array(img)
        img = torch.tensor(img).permute(2, 0, 1)
        if self.mean is not None and self.std is not None:
            img = (img - self.mean[:, None, None]) / self.std[:, None, None]

        label = Path(path).stem.upper()
        # target dtype must by int32 to work with cudnn ctc loss
        label = torch.tensor([self.stoi(s) for s in label], dtype=torch.int32)
        return img, label

    def __getitem__(self, index):
        if self.preloaded:
            return self._data[index]
        else:
            return self._read_image(self._paths[index])

    def __len__(self):
        return len(self._paths)


if __name__ == '__main__':
    data = Path('train')
    if data.is_dir():
        dataset = CaptchaDataset(data,
                                 preload=True,
                                 height=150,
                                 width=330,
                                 mean=(171.28, 181.15, 190.28),
                                 std=(94.81, 91.23, 83.4),
                                 label_size=5,
                                 )
        image, text = random.choice(dataset)
        image = np.array(image.permute(1, 2, 0))
        image = image - np.min(image, axis=(0, 1))
        image = image / np.ptp(image, axis=(0, 1)) * 255
        image = image.astype('uint8')
        image = Image.fromarray(image)
        text = ''.join(dataset.itos(s.item()) for s in text)
        print(text)
        image.show(title=text)
