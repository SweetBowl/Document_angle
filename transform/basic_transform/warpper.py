from typing import Callable, List, Union
import numpy as np

__all__ = ['Iterator', 'Selector', 'Switch', 'Mux']


class Iterator(object):
    def __init__(self, transforms: List[Callable]) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, input_data):
        output_data = input_data

        for transform in self.transforms:
            output_data = transform(output_data)

        return output_data


class Selector(object):
    def __init__(self,
                 transforms: List[Callable],
                 weights: List[Union[int, float]] = None) -> None:
        super().__init__()
        self.transforms = transforms

        if weights == None:
            self.weights = [1 / len(self.transforms)] * len(self.transforms)
        else:
            if len(weights) != len(self.transforms):
                msg = 'the length of transforms and the length of weight should be the same.'
                raise ValueError(msg)
            else:
                self.weights = weights

    def __call__(self, input_data):
        p = np.array(self.weights) / sum(self.weights)
        index = np.random.choice(len(self.transforms), replace=False, p=p)
        output_data = self.transforms[index](input_data)
        return output_data, index


class Switch(object):
    def __init__(self, transform: Callable, p: float = 1.0) -> None:
        super().__init__()
        self.transform = transform
        self.p = max(0, min(1, p))

    def __call__(self, input_data):
        output_data, flag = self.transform(input_data), True if np.random.rand(
        ) < self.p else input_data, False
        return output_data, flag


class Mux(object):
    def __init__(self, transform: Callable) -> None:
        super().__init__()
        self.transform = transform

    def __call__(self, *input_data):
        output_data = [self.transform(data) for data in input_data]
        return output_data
