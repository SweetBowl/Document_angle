from typing import Sequence, TypeVar, Union, Dict
import torch
import torch.nn as nn

__all__ = ['get_device', 'to_device']

T = TypeVar('T', torch.Tensor, torch.nn.Module)


def get_device(data: T) -> torch.device:
    '''
    Get the device of data.
    T: torch.Tensor|torch.nn.Module

    Args:
        data: data which you want to know its device.
    Return:
        torch.device
    '''
    if isinstance(data, torch.Tensor):
        return data.device
    elif isinstance(data, nn.Module):
        if isinstance(data, nn.DataParallel):
            para_dict = data.module.state_dict()
        else:
            para_dict = data.state_dict()

        keys = [para_dict.keys()]
        return para_dict[keys[0]].device
    else:
        msg = 'data should be torch.Tensorensor or torch.nn.Module, your input is {}'.format(
            type(data))
        raise ValueError(msg)


def to_device(data: Union[Sequence[T], Dict[str, T], T],
              device: torch.device) -> Union[Sequence[T], Dict[str, T], T]:
    '''
    Move data to specific device.
    T: torch.Tensor|torch.nn.Moudle

    Args:
        data: data which will be move to device.
        device: the target device of data.
    Return:
        type is the same as data. but data's device is device.
    '''
    if isinstance(data, (list, tuple)):
        return [d.to(device) for d in data]
    elif isinstance(data, dict):
        return {key: value.to(device) for key, value in data.items()}
    else:
        return data.to(device)
