from pathlib import Path
from string import ascii_uppercase

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CaptchaDataset(Dataset):
    def __init__(self, data_dir, preload,
                 ext='png',
                 mode='rgb',
                 h=130,
                 w=330,
                 label_size=None,
                 mean=(),
                 std=(),
                 ):
        super().__init__()

        self.data_dir = Path(data_dir)
        assert data_dir.is_dir(), '{} is not a valid directory'.format(data_dir)
        self.paths = list(self.data_dir.glob('*.{}'.format(ext)))
        assert len(self.paths) > 0, 'No files with {} extension found in {}'.format(ext, data_dir)

        mode_dict = {'RGB': 'RGB', 'rgb': 'RGB', 'gray': 'L', 'GRAY': 'L', 'L': 'L'}
        assert mode in mode_dict, '{} mode not supported'.format(mode)
        self.mode = mode_dict[mode]
        self.dims = 3 if mode == 'RGB' else 1

        self._symbols = list(ascii_uppercase) + [str(i) for i in range(10)]
        self._index_dict = {i: s for i, s in enumerate(self._symbols, 1)}  # 0 left for blank predicted symbol
        self._symbols_dict = {s: i for i, s in enumerate(self._symbols, 1)}

        self.preload = preload
        if self.preload:
            self._preload_data()

    def itos(self, index):
        return self._index_dict[index]

    def stoi(self, symbol):
        return self._symbols_dict[symbol]

    def _preload_data(self):
        self._data = []
        for p in self.paths:
            self._data.append(self._read_image(p))

    def _read_image(self, path):
        img = Image.open(path)
        if img.mode != self.mode:
            img.convert(self.mode)
        img = np.array(img)
        img = torch.tensor(img).permute(2, 0, 1)
        label = Path(path).stem.upper()
        # target dtype must by int32 to work with cudnn ctc loss
        label = torch.tensor([self.stoi(s) for s in label], dtype=torch.int32)
        return img, label

    def __getitem__(self, index):
        if self.preload:
            return self._data[index]

    def __len__(self):
        return len(self.paths)
