import os
import torch

__all__ = ['load_model', 'save_model']


def load_model(model: torch.nn.Module, load_dir: str, name: str):
    '''
    Load weight which is on load_dir/name to model.

    Args:
        model(torch.nn.Module)
        load_dir(str)
        name(str)
    '''
    weight_load_path = os.path.join(load_dir, name)

    if not os.path.exists(weight_load_path):
        msg = 'this file {} doesn\'t exist'.format(weight_load_path)
        raise FileExistsError(msg)

    model.load_state_dict(torch.load(weight_load_path, map_location='cpu'))


def save_model(model: torch.nn.Module, save_dir: str, name: str):
    '''
    Save model's weight on save_dir/name.

    Args:
        model(torch.nn.Module)
        save_dir(str)
        name(str)
    '''
    weight_save_path = os.path.join(save_dir, name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if isinstance(model, torch.nn.Module):
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), weight_save_path)
        else:
            torch.save(model.state_dict(), weight_save_path)
    else:
        msg = 'the variable model should be torch.nn.Module, but now is {}'.format(
            type(model))
        raise ValueError(msg)
