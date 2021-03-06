import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from .basic_transform import ScaleResize, Selector
from .abstract_transform import AbstractTransform
from typing import Dict, Any
import torch

__all__ = ['ImageTrainTransform', 'ImageTestTransform']


class ImageTrainTransform(AbstractTransform):
    def __init__(self, fixed_size) -> None:
        super().__init__()
        self.resize = ScaleResize(fixed_size=fixed_size,
                                  fill_value=(255, 255, 255))
        self.to_tensor = transforms.ToTensor()
        self.gray_scale = transforms.Grayscale(1)

        self.small_angle_rotate = transforms.RandomRotation(degrees=2)
        self.colorjitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        transformed_dict = {}
        image = input_dict['image']
        image = self.resize(image)
        image = self.colorjitter(image)

        # change the image channels from 3 to 1
        image = self.gray_scale(image)

        image = self.small_angle_rotate(image)
        image = self.to_tensor(image)
        rotated_imgs = [
            F.rotate(image, 0), F.rotate(image, 90), F.rotate(image, 180), F.rotate(image, 270)
        ]
        rotated_imgs = torch.stack(rotated_imgs, dim=0)
        # image, rotate_flag = self.rotate(image)

        rotate_flag = torch.LongTensor([0, 1, 2, 3])
        transformed_dict['image'] = rotated_imgs
        transformed_dict['rotate_flag'] = rotate_flag
        # return image
        return transformed_dict


class ImageTestTransform(AbstractTransform):
    def __init__(self, fixed_size) -> None:
        super().__init__()
        self.resize = ScaleResize(fixed_size=fixed_size,
                                  fill_value=(255, 255, 255))
        self.to_tensor = transforms.ToTensor()
        self.fixed_size = fixed_size

        self.gray_scale = transforms.Grayscale(1)

    def __call__(self, input_dict: Dict[str, Any]):
        # -> Dict[str, Any]

        transformed_dict = {}
        image = input_dict['image']
        # image = self.resize(image)

        # change image channels from 3 to 1
        image = self.gray_scale(image)

        # return the 4 rotated copies of the image and the flag of the rotation
        # i.e. 0 for 0 degrees, 1 for 90 degrees, 2 for 180 degrees, 3 for 270 degrees
        image = self.to_tensor(image)

        rotated_imgs = [
            F.rotate(image, 0), F.rotate(image, 90), F.rotate(image, 180), F.rotate(image, 270)
        ]

        rotated_imgs = torch.stack(rotated_imgs, dim=0)

        rotate_flags = torch.LongTensor([0, 1, 2, 3])
        # [4,1]
        # return [1,3,1600]...
        # rotate_flags = rotate_flags.view([-1,1])

        transformed_dict['image'] = rotated_imgs

        transformed_dict['rotate_flag'] = rotate_flags
        # return (rotated_imgs,rotate_flags)
        return transformed_dict


class ImageTestTransformOneFix(AbstractTransform):
    def __init__(self,fixed_size) -> None:
        super().__init__()
        self.resize = ScaleResize(fixed_size=fixed_size,
                                  fill_value=(255, 255, 255))
        self.to_tensor = transforms.ToTensor()
        rotate_fn = [
            lambda img: F.rotate(img, 0), lambda img: F.rotate(img, 90),
            lambda img: F.rotate(img, 180), lambda img: F.rotate(img, 270)
        ]
        self.rotate = Selector(transforms=rotate_fn)
        # self.small_angle_rotate = transforms.RandomRotation(degrees=5)
        # self.colorjitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        self.gray_scale = transforms.Grayscale(1)

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        transformed_dict = {}
        image = input_dict['image']
        image = self.resize(image)
        image, rotate_flag = self.rotate(image)
        # image = self.small_angle_rotate(image)
        image = self.gray_scale(image)
        image = self.to_tensor(image)
        transformed_dict['image'] = image
        rotate_flag = torch.LongTensor([rotate_flag])
        transformed_dict['rotate_flag'] = rotate_flag
        return transformed_dict

class ImageTestTransformOneRaw(AbstractTransform):
    def __init__(self):
        super().__init__()
        rotate_fn = [
            lambda img: F.rotate(img, 0), lambda img: F.rotate(img, 90),
            lambda img: F.rotate(img, 180), lambda img: F.rotate(img, 270)
        ]
        self.resize = ScaleResize(fixed_size=(1000,1000),fill_value=(255,255,255))
        self.rotate = Selector(transforms=rotate_fn)
        self.to_tensor = transforms.ToTensor()
        self.gray_scale = transforms.Grayscale(1)

    def __call__(self,image):
        image = self.resize(image)
        image, rotate_flag = self.rotate(image)
        image = self.gray_scale(image)
        image = self.to_tensor(image)
        rotate_flag = torch.LongTensor([rotate_flag])
        return (image, rotate_flag)

