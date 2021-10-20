import torch.utils.data as data

__all__ = ['generate_data_loaders']


def generate_data_loaders(datasets):
    '''
    Put dataset into corresponding data loader.

    Type `DS`: Union[data.dataset.Dataset, data.ConcatDataset, dataset.AbstractDataset]
    Type `DL`: data.dataloader.DataLoader

    Args:
        datasets(Dict[str, Union[DS, Sequence[DS]]): A dict containing datasets, its keys can be train, val or test. If dataset in datasets is not instance of AbstractDataset, it should have these attributes: batch_size, shuffle, num_workers, prefetch_factor, collate_fn.
    Returns:
        (Dict[str, Union[DL, Sequence[DL]]): A dict containing data loaders, its keys are the same as datasets.
    '''
    data_loaders = {}

    for key, datasets in datasets.items():
        if isinstance(datasets, (tuple, list)):
            data_loaders[key] = [
                _generate_dataloader(dataset) for dataset in datasets
            ]
        else:
            data_loaders[key] = _generate_dataloader(datasets)

    return data_loaders


def _generate_dataloader(dataset):
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=dataset.shuffle,
                                  num_workers=dataset.num_workers,
                                  prefetch_factor=dataset.prefetch_factor,
                                  collate_fn=dataset.collate_fn)
    return data_loader
