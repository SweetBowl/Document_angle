import math

__all__ = ['calc_padding_size']


def calc_padding_size(target_size,
                      original_size,
                      height_align='center',
                      width_align='center'):
    '''
    Calculate the padding size of origin size to target size.

    Args:
        target_size(Tuple[int]): The targeted size of padded image and its shape should be (h, w).
        original_size(Tuple[int]): The original size of image which need padding and its shape should be (h, w).
        height_align(str): The alignment in the vertical direction, its value should be center, top or bottom.
        width_align(str): The alignment in the horizontal direction, its value should be center, left or right.
    Returns:
        (Tuple[int]): Padding size in each direction. (left, top, right, bottom)
    '''
    if not isinstance(target_size,
                      tuple) and len(target_size) != 2 and not isinstance(
                          (target_size[0], target_size[1]), int):
        msg = 'target_size should be a tuple and shape is (h,w), but got {}'.format(
            target_size)
        raise ValueError(msg)

    if not isinstance(original_size,
                      tuple) and len(original_size) != 2 and not isinstance(
                          (original_size[0], original_size[1]), int):
        msg = 'original_size should be a tuple and shape is (h,w), but got {}'.format(
            original_size)
        raise ValueError(msg)

    if height_align not in ['center', 'top', 'bottom']:
        msg = 'heigth_align should be center, top or bottom, but got {}'.format(
            height_align)
        raise ValueError(msg)

    if width_align not in ['center', 'left', 'right']:
        msg = 'width_align should be center, left or right, but got {}'.format(
            width_align)
        raise ValueError(msg)

    target_h, target_w = target_size
    original_h, original_w = original_size
    padding_height = target_h - original_h
    padding_width = target_w - original_w

    if height_align == 'center':
        padding_top = math.floor(padding_height // 2)
    elif height_align == 'top':
        padding_top = 0
    else:
        padding_top = padding_height

    if width_align == 'center':
        padding_left = math.floor(padding_width // 2)
    elif width_align == 'left':
        padding_left = 0
    else:
        padding_left = padding_width

    padding_right = padding_width - padding_left
    padding_bottom = padding_height - padding_top
    return (padding_left, padding_top, padding_right, padding_bottom)
