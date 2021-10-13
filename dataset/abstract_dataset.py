import torch.utils.data as data
import torch
from abc import abstractclassmethod


class AbstractDataset(data.Dataset):
    '''
    Abstract Base Class for Dataset. All Dataset class will inherit from this class.
    '''
    def __init__(self, data_frame, shuffle=False, transforms={}) -> None:
        super().__init__()
        self.data_frame = data_frame
        self._shuffle = shuffle
        self.transforms = transforms
        self._keys = list(self.data_frame.keys())
        self.collate_fn = None

    def __len__(self):
        return len(self.data_frame)

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        if value not in [True, False]:
            msg = 'shuffle should be True or False.'
            raise ValueError(msg)

        self._shuffle = value

    @property
    def keys(self):
        return self._keys

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist(index)

        info_dict = self.data_frame.loc[index, :]
        item_dict = {'index': index}

        for key in self.keys:
            item_dict[key] = info_dict[key]

        return item_dict
