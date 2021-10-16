import torchvision.transforms.functional as F
from PIL import Image
import torch
import math
import utils

__all__ = ['ScaleResize']


class ScaleResize(object):
    def __init__(self,
                 fixed_size,
                 fill_value=0,
                 interpolation=Image.BILINEAR) -> None:
        '''
        Resize Image and keep the aspect ratio.

        Args:
            fixed_size(Tuple[int]): The target size of resized image and its shape should be (h,w).
            fill_value(Union[float, Tuple[float]]): The value filled in blank area.
            interpolation(PIL.Image.InterpolationMode): Desired interpolation method defined in PIL.Image.
        '''
        super().__init__()

        if not isinstance(fixed_size,
                          tuple) and len(fixed_size) != 2 and not isinstance(
                              (fixed_size[0], fixed_size[1]), int):
            msg = 'fixed_size should be tuple and shape is (h,w), but got {}'.format(
                fixed_size)
            raise ValueError(msg)
        else:
            self.fixed_size = fixed_size

        self.fill_value = fill_value
        self.interpolation = interpolation

    def __call__(self, image):
        '''
        Args:
            image(Union[PIL.Image.Image, torch.Tensor]): Image to resize and its aspect ratio will be unchanged.
        Returns:
            (Union[PIL.Image.Image, torch.Tensor]): Padded image.
        '''
        if isinstance(image, Image.Image):
            w = image.width
            h = image.height
        elif isinstance(image, torch.Tensor):
            c, h, w = image.size()
        else:
            msg = 'image shoude be PIL.Image or torch.Tensor, but got {}.'.format(
                type(image))
            raise ValueError(msg)

        h_scale = h / self.fixed_size[0]
        w_scale = w / self.fixed_size[1]

        if h_scale >= w_scale:
            new_h = self.fixed_size[0]
            new_w = math.floor(1 / h_scale * w)
        else:
            new_w = self.fixed_size[1]
            new_h = math.floor(1 / w_scale * h)

        resized_image = F.resize(image,
                                 size=(new_h, new_w),
                                 interpolation=self.interpolation)

        padding_left, padding_top, padding_right, padding_bottom = utils.calc_padding_size(
            self.fixed_size, (new_h, new_w))
        padded_image = F.pad(
            resized_image,
            padding=[padding_left, padding_top, padding_right, padding_bottom],
            fill=self.fill_value)
        return padded_image
