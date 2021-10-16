import torch.utils.data as data
import torch


class AbstractDataset(data.Dataset):
    '''
    Abstract Base Class for Dataset. All Dataset class will inherit from this class.
    '''
    def __init__(self,
                 data_frame,
                 transform=None,
                 batch_size=1,
                 shuffle=False,
                 num_workers=1,
                 prefetch_factor=2,
                 collate_fn=None) -> None:
        '''
        Args:
            data_frame(pandas.DataFrame)
            transform(Callable)
            batch_size(int)
            shuffle(bool)
            num_workers(int)
            prefetch_factor(int)
            collate_fn(Callable)
        '''
        super().__init__()
        self.data_frame = data_frame
        self.transform = transform
        self._keys = list(self.data_frame.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.data_frame)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if not isinstance(value, int):
            msg = 'batch_size should be Type int, but got {}.'.format(
                type(value))
            raise TypeError(msg)

        if value > len(self):
            msg = 'batch_size should be less than the size of dataset({}), but got {}.'.format(
                len(self), value)
            raise ValueError(msg)

        self._batch_size = value

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        if value not in [True, False]:
            msg = 'shuffle should be Type bool, but got {}.'.format(
                type(value))
            raise TypeError(msg)

        self._shuffle = value

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        if not isinstance(value, int):
            msg = 'num_workers should be Type int, but got {}.'.format(
                type(value))
            raise TypeError(msg)

        self._num_workers = value

    @property
    def prefetch_factor(self):
        return self._prefetch_factor

    @prefetch_factor.setter
    def prefetch_factor(self, value):
        if not isinstance(value, int):
            msg = 'prefetch_factor should be Type int, but got {}.'.format(
                type(value))
            raise TypeError(msg)

        self._prefetch_factor = value

    @property
    def collate_fn(self):
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        if not hasattr(value, '__all__') and value is not None:
            msg = 'collate_fn should be Type Callable or NoneType, but got {}.'.format(
                type(value))
            raise TypeError(msg)

        self._collate_fn = value

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
