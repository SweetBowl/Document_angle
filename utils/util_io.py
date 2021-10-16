import os
import torch

__all__ = ['load_model', 'save_model']


def load_model(model, load_path):
    '''
    Load weight from file which is on load_path to model.

    Args:
        model(torch.nn.Module)
        load_path(str): the path os model weight file.
    Returns:
        (torch.nn.Module): A model loaded with weight.
    '''
    if not os.path.exists(load_path):
        msg = 'this file {} doesn\'t exist'.format(load_path)
        raise FileExistsError(msg)

    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    return model


def save_model(model, save_path):
    '''
    Save model's weight on save_path.

    Args:
        model(torch.nn.Module)
        save_path(str): The path of saved model weight file.
    '''
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(model, torch.nn.Module):
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
    else:
        msg = 'model should be torch.nn.Module, but got {}'.format(type(model))
        raise TypeError(msg)
