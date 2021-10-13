import torch.utils.data as data

__all__ = ['generate_data_loaders']


def generate_data_loaders(datasets, config):
    '''
    Put dataset into corresponding data loader. The return value have the same structure as the input parameter

    Args:
        datasets: dict(str:torch.utils.data.Dataset | dict(src:torch.utils.data.Dataset, ...), ...)
        config: config.AbstractConfig
    '''
    data_loaders = {}

    if 'train' in datasets.keys():
        train_datasets = datasets['train']

        if isinstance(train_datasets, dict):
            data_loaders['train'] = {}

            for dataset_name in train_datasets.keys():
                data_loaders['train'][dataset_name] = _generate_dataloader(
                    train_datasets[dataset_name], config)
        else:
            data_loaders['train'] = _generate_dataloader(
                train_datasets, config)

    if 'val' in datasets.keys():
        val_datasets = datasets['val']

        if isinstance(val_datasets, dict):
            data_loaders['val'] = {}

            for dataset_name in val_datasets.keys():
                data_loaders['val'][dataset_name] = _generate_dataloader(
                    val_datasets[dataset_name], config)
        else:
            data_loaders['val'] = _generate_dataloader(val_datasets, config)

    if 'test' in datasets.keys():
        test_datasets = datasets['test']

        if isinstance(test_datasets, dict):
            data_loaders['test'] = {}

            for dataset_name in test_datasets.keys():
                data_loaders['test'][dataset_name] = _generate_dataloader(
                    test_datasets[dataset_name], config)
        else:
            data_loaders['test'] = _generate_dataloader(test_datasets, config)

    return data_loaders


def _generate_dataloader(dataset, cfg):
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=cfg.BATCH_SIZE,
                                  shuffle=dataset.shuffle,
                                  num_workers=cfg.WORKER,
                                  prefetch_factor=cfg.PREFETCH_FACTOR,
                                  collate_fn=dataset.collate_fn)
    return data_loader
