from os import path
from .abstract_dataset import *
from PIL import Image
import pandas as pd
from transform import ImageTrainTransform, ImageTestTransform

__all__ = [
    'Bank_Train', 'Bank_Val', 'Bank_Test', 'Doc_Train', 'Doc_Val', 'Doc_Test'
]


class ImageDataset(AbstractDataset):
    def __init__(self,
                 data_frame,
                 transform=None,
                 batch_size=1,
                 shuffle=False,
                 num_workers=1) -> None:
        super().__init__(data_frame,
                         transform=transform,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers)

    def __getitem__(self, index):
        item_dict = super().__getitem__(index)
        item_dict['image'] = Image.open(item_dict['image_path']).convert('RGB')
        data = self.transform(item_dict)
        return data


def Bank_Train(cfg):
    csv_path = path.join(path.dirname(__file__), '..', 'Data/Bank_Train.csv')
    data_frame = pd.read_csv(csv_path)
    return ImageDataset(
        data_frame=data_frame,
        transform=ImageTrainTransform(fixed_size=cfg.IMAGE_SIZE),
        shuffle=True,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        prefetch_factor=cfg.PREFETCH_FACTOR,
    )


def Bank_Val(cfg):
    csv_path = path.join(path.dirname(__file__), '..', 'Data/Bank_Val.csv')
    data_frame = pd.read_csv(csv_path)
    return ImageDataset(
        data_frame=data_frame,
        transform=ImageTestTransform(fixed_size=cfg.IMAGE_SIZE),
        shuffle=False,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        prefetch_factor=cfg.PREFETCH_FACTOR,
    )


def Bank_Test(cfg):
    csv_path = path.join(path.dirname(__file__), '..', 'Data/Bank_Test.csv')
    data_frame = pd.read_csv(csv_path)
    return ImageDataset(
        data_frame=data_frame,
        transform=ImageTestTransform(fixed_size=cfg.IMAGE_SIZE),
        shuffle=False,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        prefetch_factor=cfg.PREFETCH_FACTOR)


def Doc_Train(cfg):
    csv_path = path.join(path.dirname(__file__), '..', 'Data/Doc_Train.csv')
    data_frame = pd.read_csv(csv_path)
    return ImageDataset(
        data_frame=data_frame,
        transform=ImageTrainTransform(fixed_size=cfg.IMAGE_SIZE),
        shuffle=True,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        prefetch_factor=cfg.PREFETCH_FACTOR,
    )


def Doc_Val(cfg):
    csv_path = path.join(path.dirname(__file__), '..', 'Data/Doc_Val.csv')
    data_frame = pd.read_csv(csv_path)
    return ImageDataset(
        data_frame=data_frame,
        transform=ImageTestTransform(fixed_size=cfg.IMAGE_SIZE),
        shuffle=False,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        prefetch_factor=cfg.PREFETCH_FACTOR,
    )


def Doc_Test(cfg):
    csv_path = path.join(path.dirname(__file__), '..', 'Data/Bank_Test.csv')
    data_frame = pd.read_csv(csv_path)
    return ImageDataset(
        data_frame=data_frame,
        transform=ImageTestTransform(fixed_size=cfg.IMAGE_SIZE),
        shuffle=False,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        prefetch_factor=cfg.PREFETCH_FACTOR,
    )
