import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from .basic_transform import ScaleResize, Selector
from .abstract_transform import AbstractTransform
from typing import Dict, Any
import torch

__all__ = ['ImageTrainTransform', 'ImageTestTransform']


class ImageTestTransform(AbstractTransform):
    def __init__(self, fixed_size) -> None:
        super().__init__()
        self.resize = ScaleResize(fixed_size=fixed_size,
                                  fill_value=(255, 255, 255))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        transformed_dict = {}
        image = input_dict['image']
        image = self.resize(image)

        # return the 4 rotated copies of the image and the flag of the rotation
        # i.e. 0 for 0 degrees, 1 for 90 degrees, 2 for 180 degrees, 3 for 270 degrees
        image = self.to_tensor(image)

        rotated_imgs = [
            F.rotate(image, 0), F.rotate(image, 90), F.rotate(image, 180), F.rotate(image, 270)
        ]

        rotated_imgs = torch.stack(rotated_imgs, dim=0)
        rotate_flags = torch.LongTensor([0,1,2,3])

        transformed_dict['image'] = rotated_imgs

        transformed_dict['rotate_flag'] = rotate_flags
        return transformed_dict

class ImageTrainTransform(AbstractTransform):
    def __init__(self, fixed_size) -> None:
        super().__init__()
        self.resize = ScaleResize(fixed_size=fixed_size,
                                  fill_value=(255, 255, 255))
        self.to_tensor = transforms.ToTensor()
        rotate_fn = [
            lambda img: F.rotate(img, 0), lambda img: F.rotate(img, 90),
            lambda img: F.rotate(img, 180), lambda img: F.rotate(img, 270)
        ]
        self.rotate = Selector(transforms=rotate_fn)
        self.small_angle_rotate = transforms.RandomRotation(degrees=5)
        self.colorjitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        transformed_dict = {}
        image = input_dict['image']
        image = self.resize(image)
        image, rotate_flag = self.rotate(image)
        image = self.small_angle_rotate(image)
        image = self.to_tensor(image)
        transformed_dict['image'] = image
        rotate_flag = torch.LongTensor([rotate_flag])
        transformed_dict['rotate_flag'] = rotate_flag
        return transformed_dict