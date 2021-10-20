import torch
from abc import ABC, abstractclassmethod


class AbstractStep(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractclassmethod
    def load_data(self) -> dict(torch.utils.data.DataLoader):
        '''
        Load dataset.
        '''
        raise NotImplementedError(
            'load_data should be implemented by subclass.')

    @abstractclassmethod
    def load_models(self):
        '''
        Load models.
        '''
        raise NotImplementedError(
            'load_model should be implemented by subclass.')

    @abstractclassmethod
    def load_optimizers(self):
        '''
        Load optimizers.
        '''
        raise NotImplementedError(
            'load_optimizers should be implemented by subclass.')

    @abstractclassmethod
    def load_criterions(self):
        '''
        load criterions.
        '''
        raise NotImplementedError(
            'load_criterions should be implemented by subclass.')

    @abstractclassmethod
    def load_meters(self):
        '''
        load meters.
        '''
        raise NotImplementedError(
            'load_meters should be implemented by subclass.')

    @abstractclassmethod
    def forward(self, input_data):
        '''
        forward inference.
        '''
        raise NotImplementedError('forward should be implemented by subclass.')

    @abstractclassmethod
    def backward(self, input_data, output_data):
        '''
        backward propagation.
        '''
        raise NotImplementedError(
            'backward should be implemented by subclass.')

    @property
    def data_loaders(self):
        return self._data_loaders

    @property
    def models(self):
        return self._models

    @property
    def criterions(self):
        return self._criterions

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def meters(self):
        return self._meters
